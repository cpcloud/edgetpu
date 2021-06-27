use crate::{
    error::{check_null, Error},
    tflite_sys,
};
#[cfg(feature = "posenet_decoder")]
use ndarray::{ArrayView, IntoDimension};
use num_traits::ToPrimitive;
use std::{convert::TryFrom, ffi::CStr, marker::PhantomData};

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

    pub(crate) fn name(&self) -> Result<&str, Error> {
        unsafe { CStr::from_ptr((*self.tensor).name) }
            .to_str()
            .map_err(Error::GetTensorName)
    }

    pub(crate) fn r#type(&self) -> tflite_sys::TfLiteType {
        unsafe { (*self.tensor).type_ }
    }

    pub(crate) fn quantization_params(&self) -> tflite_sys::TfLiteQuantizationParams {
        unsafe { (*self.tensor).params }
    }

    /// View a tensor's data as a slice of `T` values.
    ///
    /// # Safety
    ///
    /// The `self.tensor.data.raw` member must point to a valid array of `T`.
    #[cfg(feature = "posenet_decoder")]
    pub(crate) fn as_f32_slice(&'interp self) -> Result<&'interp [f32], Error> {
        let typ = self.r#type();
        if typ != tflite_sys::TfLiteType::kTfLiteFloat32 {
            return Err(Error::GetTensorSlice(typ));
        }
        Ok(unsafe { std::slice::from_raw_parts((*self.tensor).data.f, self.len) })
    }

    /// Mutable view of a tensor's data as a slice of `T` values.
    pub(crate) fn as_u8_slice(&'interp mut self) -> Result<&'interp [u8], Error> {
        let typ = self.r#type();
        if typ != tflite_sys::TfLiteType::kTfLiteUInt8 {
            return Err(Error::GetTensorSlice(typ));
        }
        Ok(unsafe { std::slice::from_raw_parts((*self.tensor).data.uint8, self.len) })
    }

    #[cfg(feature = "posenet_decoder")]
    pub(crate) fn as_ndarray<T, I>(
        &'interp self,
        slice: &'interp [T],
        dims: I,
    ) -> Result<ArrayView<'interp, T, I::Dim>, Error>
    where
        I: IntoDimension,
    {
        ArrayView::from_shape(dims.into_dimension(), slice).map_err(Error::ConstructArrayView)
    }

    #[inline]
    pub(crate) fn dequantized(&'interp mut self) -> Result<Vec<f32>, Error> {
        self.dequantized_with_scale(1.0)
    }

    pub(crate) fn dequantized_with_scale(
        &'interp mut self,
        mut scale: f32,
    ) -> Result<Vec<f32>, Error> {
        let tflite_sys::TfLiteQuantizationParams {
            zero_point,
            scale: quant_scale,
        } = self.quantization_params();
        scale *= quant_scale;
        let zero_point = zero_point.to_f32().ok_or(Error::ConvertToF32)?;
        Ok(self
            .as_u8_slice()?
            .iter()
            .map(|&value| (f32::from(value) - zero_point) * scale)
            .collect())
    }

    pub(crate) fn dim(&self, index: usize) -> Result<usize, Error> {
        dim(self.tensor, index)
    }
}
