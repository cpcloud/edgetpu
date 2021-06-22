use crate::{
    error::{check_null, tflite_status_to_result, Error},
    tflite_sys,
};
use ndarray::{Array, ArrayView, IntoDimension};
use num_traits::ToPrimitive;
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

    fn quantization_params(&self) -> tflite_sys::TfLiteQuantizationParams {
        unsafe { (*self.tensor).params }
    }

    /// View a tensor's data as a slice of `T` values.
    ///
    /// # Safety
    ///
    /// The `self.tensor.data.raw` member must point to a valid array of `T`.
    #[cfg(feature = "posenet_decoder")]
    pub(crate) unsafe fn as_slice<T>(&'interp self) -> &'interp [T]
    where
        T: Sized + Copy,
    {
        std::slice::from_raw_parts((*self.tensor).data.raw.cast::<T>(), self.len)
    }

    #[cfg(feature = "posenet_decoder")]
    pub(crate) fn as_ndarray<T, I>(
        &'interp self,
        slice: &'interp [T],
        dims: I,
    ) -> Result<ArrayView<'interp, T, I::Dim>, Error>
    where
        T: Sized + Copy,
        I: IntoDimension,
    {
        ArrayView::from_shape(dims.into_dimension(), slice).map_err(Error::ConstructArrayView)
    }

    pub(crate) fn as_ndarray_dequantized<T, I>(
        &'interp self,
        slice: &'interp [T],
        dims: I,
    ) -> Result<Array<f32, I::Dim>, Error>
    where
        T: Into<f32> + Clone + Sized + Copy,
        I: IntoDimension,
    {
        let tflite_sys::TfLiteQuantizationParams { scale, zero_point } = self.quantization_params();
        let zero_point = zero_point
            .to_f32()
            .expect("failed to convert zero point to f32");
        Ok(ArrayView::from_shape(dims.into_dimension(), slice)
            .map_err(Error::ConstructArrayView)?
            .mapv(|v| (v.into() - zero_point) / scale))
    }

    #[cfg(feature = "posenet_decoder")]
    pub(crate) fn dim(&self, index: usize) -> Result<usize, Error> {
        dim(self.tensor, index)
    }

    pub(crate) fn copy_from_buffer(&mut self, buf: &[u8]) -> Result<(), Error> {
        tflite_status_to_result(
            // SAFETY: buf is guaranteed to be valid and have length buf.len()
            unsafe {
                tflite_sys::TfLiteTensorCopyFromBuffer(
                    self.tensor as _,
                    buf.as_ptr().cast(),
                    buf.len(),
                )
            },
            "failed to copy from input buffer",
        )
    }
}
