use crate::{
    error::{check_null, tflite_status_to_result, Error},
    tflite_sys,
};
use ndarray::{ArrayView, IntoDimension};
use std::{convert::TryFrom, marker::PhantomData};

fn dim(tensor: *const tflite_sys::TfLiteTensor, index: usize) -> Result<usize, Error> {
    assert!(!tensor.is_null());
    let dims = num_dims(tensor)?;
    if index >= dims {
        return Err(Error::GetDim(index, dims));
    }

    let index = i32::try_from(index).map_err(Error::ConvertUSizeToI32Index)?;
    // # SAFETY: self.tensor is guaranteed to be valid
    usize::try_from(unsafe { tflite_sys::TfLiteTensorDim(tensor, index) })
        .map_err(Error::ConvertDimI32ToUSize)
}

fn num_dims(tensor: *const tflite_sys::TfLiteTensor) -> Result<usize, Error> {
    assert!(!tensor.is_null());
    // # SAFETY: self.tensor is guaranteed to be valid
    usize::try_from(unsafe { tflite_sys::TfLiteTensorNumDims(tensor) }).map_err(Error::GetNumDims)
}

/// A safe wrapper around TfLiteTensor.
pub(crate) struct Tensor<'interp> {
    tensor: *const tflite_sys::TfLiteTensor,
    len: usize,
    // Data are owned by the interpreter that allocated the tensor.
    _p: PhantomData<&'interp ()>,
}

impl<'interp> Tensor<'interp> {
    pub(super) fn new(tensor: *const tflite_sys::TfLiteTensor) -> Result<Self, Error> {
        let tensor = check_null(tensor).ok_or(Error::CreateTensor)?;
        Ok(Self {
            tensor,
            len: (0..num_dims(tensor)?).try_fold(1, |size, d| Ok(size * dim(tensor, d)?))?,
            _p: Default::default(),
        })
    }

    pub(crate) fn as_slice(&'interp self) -> &'interp [f32] {
        unsafe { std::slice::from_raw_parts((*self.tensor).data.f, self.len) }
    }

    pub(crate) fn as_ndarray<I>(
        &'interp self,
        dims: I,
    ) -> Result<ArrayView<'interp, f32, I::Dim>, Error>
    where
        I: IntoDimension,
    {
        ArrayView::from_shape(dims.into_dimension(), self.as_slice())
            .map_err(Error::ConstructArrayView)
    }

    #[cfg(feature = "posenet_decoder")]
    pub(crate) fn dim(&self, index: usize) -> Result<usize, Error> {
        dim(self.tensor, index)
    }

    pub(crate) fn copy_from_raw_buffer(&mut self, buf: *const u8, len: usize) -> Result<(), Error> {
        assert!(!buf.is_null());
        // SAFETY: buf is guaranteed to be valid and of len buf.len()
        tflite_status_to_result(
            unsafe { tflite_sys::TfLiteTensorCopyFromBuffer(self.tensor as _, buf.cast(), len) },
            "failed to copy from input buffer",
        )
    }
}
