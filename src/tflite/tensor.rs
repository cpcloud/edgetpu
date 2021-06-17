use crate::{
    error::{tflite_status_to_result, Error},
    tflite_sys,
};
use ndarray::{ArrayView, IntoDimension};
use std::{convert::TryFrom, marker::PhantomData};

fn dim(tensor: *const tflite_sys::TfLiteTensor, dim_index: usize) -> usize {
    // # SAFETY: self.tensor is guaranteed to be valid
    usize::try_from(unsafe {
        tflite_sys::TfLiteTensorDim(
            tensor,
            i32::try_from(dim_index).expect("failed to convert dim_index usize to i32"),
        )
    })
    .expect("failed to convert dim i32 to usize")
}

fn num_dims(tensor: *const tflite_sys::TfLiteTensor) -> usize {
    // # SAFETY: self.tensor is guaranteed to be valid
    usize::try_from(unsafe { tflite_sys::TfLiteTensorNumDims(tensor) })
        .expect("failed to convert dim i32 to usize")
}

pub(crate) struct Tensor<'a> {
    tensor: *const tflite_sys::TfLiteTensor,
    len: usize,
    _p: PhantomData<&'a ()>,
}

impl<'a> Tensor<'a> {
    pub(super) fn new(tensor: *const tflite_sys::TfLiteTensor) -> Result<Self, Error> {
        Ok(Self {
            tensor,
            len: (0..num_dims(tensor as _)).fold(1, |size, d| size * dim(tensor as _, d)),
            _p: Default::default(),
        })
    }

    pub(crate) fn as_slice(&'a self) -> &'a [f32] {
        unsafe { std::slice::from_raw_parts((*self.tensor).data.f, self.len) }
    }

    pub(crate) fn as_ndarray<I>(&'a self, dims: I) -> Result<ArrayView<'a, f32, I::Dim>, Error>
    where
        I: IntoDimension,
    {
        ArrayView::from_shape(dims.into_dimension(), self.as_slice())
            .map_err(Error::ConstructArrayView)
    }

    pub(crate) fn dim(&self, dim_index: usize) -> usize {
        dim(self.tensor as _, dim_index)
    }

    pub(crate) fn copy_from_buffer(&mut self, buf: &[u8]) -> Result<(), Error> {
        // SAFETY: buf is guaranteed to be valid and of len buf.len()
        tflite_status_to_result(
            unsafe {
                tflite_sys::TfLiteTensorCopyFromBuffer(
                    self.tensor as _,
                    buf.as_ptr() as _,
                    buf.len(),
                )
            },
            "failed to copy from input buffer",
        )
    }
}
