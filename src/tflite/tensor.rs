use crate::{
    error::{check_null, Error},
    ffi::ffi,
    tflite_sys,
};
use std::{convert::TryFrom, ffi::CStr, marker::PhantomData};

fn dim(tensor: *const tflite_sys::TfLiteTensor, index: usize) -> Result<usize, Error> {
    let dims = num_dims(check_null(tensor).ok_or(Error::GetTensorForDim)?)?;
    if index >= dims {
        return Err(Error::GetDim(index, dims));
    }

    let index = i32::try_from(index).map_err(Error::ConvertUSizeToI32Index)?;
    // # SAFETY: self.tensor is guaranteed to be valid
    usize::try_from(unsafe { tflite_sys::TfLiteTensorDim(tensor, index) })
        .map_err(Error::ConvertDimI32ToUSize)
}

fn num_dims(tensor: *const tflite_sys::TfLiteTensor) -> Result<usize, Error> {
    // # SAFETY: self.tensor is guaranteed to be valid
    usize::try_from(unsafe {
        tflite_sys::TfLiteTensorNumDims(check_null(tensor).ok_or(Error::GetTensorForNumDims)?)
    })
    .map_err(Error::GetNumDims)
}

/// A safe wrapper around TfLiteTensor.
pub(crate) struct Tensor<'interp> {
    tensor: *const tflite_sys::TfLiteTensor,
    // Data are owned by the interpreter that allocated the tensor.
    _p: PhantomData<&'interp ()>,
}

impl<'interp> Tensor<'interp> {
    pub(super) fn new(tensor: *const tflite_sys::TfLiteTensor) -> Result<Self, Error> {
        let tensor = check_null(tensor).ok_or(Error::CreateTensor)?;
        Ok(Self {
            tensor,
            _p: Default::default(),
        })
    }

    pub(crate) fn name(&self) -> Result<&str, Error> {
        unsafe { CStr::from_ptr((*self.tensor).name) }
            .to_str()
            .map_err(Error::GetTensorName)
    }

    pub(crate) fn dequantize_with_scale(&'interp self, scale: f32) -> Result<Vec<f32>, Error> {
        unsafe {
            ffi::dequantize_with_scale(self.tensor.as_ref().expect("self.tensor is null"), scale)
                .map_err(Error::Dequantize)
        }
    }

    pub(crate) fn dim(&self, index: usize) -> Result<usize, Error> {
        dim(self.tensor, index)
    }
}
