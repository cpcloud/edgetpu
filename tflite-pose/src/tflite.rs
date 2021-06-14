use crate::{error::Error, tflite_sys};
use std::{convert::TryFrom, ffi::CString, marker::PhantomData, path::Path};

pub(super) struct Model {
    model: *mut tflite_sys::TfLiteModel,
}

#[cfg(unix)]
pub fn path_to_c_string<P>(path: P) -> Result<CString, Error>
where
    P: AsRef<Path>,
{
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_ref().as_os_str().as_bytes()).map_err(Error::PathToCString)
}

#[cfg(windows)]
pub fn path_to_c_string<P>(path: P) -> Result<CString, Error>
where
    P: AsRef<Path>,
{
    CString::new(path.as_ref().to_str().ok_or(Error::PathToStr)?).map_err(Error::PathToCString)
}

impl Model {
    pub(super) fn new<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let path_ref = path.as_ref();
        let path_bytes = path_to_c_string(path_ref)?;
        Ok(Self {
            model: unsafe { tflite_sys::TfLiteModelCreateFromFile(path_bytes.as_ptr()) },
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            tflite_sys::TfLiteModelDelete(self.model);
        }
    }
}

pub(crate) struct Tensor<'a> {
    tensor: *mut tflite_sys::TfLiteTensor,
    _p: PhantomData<&'a ()>,
}

impl<'a> Tensor<'a> {
    fn new(tensor: *mut tflite_sys::TfLiteTensor) -> Result<Self, Error> {
        Ok(Self {
            tensor,
            _p: PhantomData,
        })
    }

    pub(crate) fn as_slice(&'a self) -> &'a [f32] {
        let num_bytes = self.byte_size();
        unsafe { std::slice::from_raw_parts((*self.tensor).data.f, num_bytes) }
    }

    pub(crate) fn as_ndarray<D: ndarray::Dimension>(
        &'a self,
        dims: D,
    ) -> Result<ndarray::ArrayView<'a, f32, D>, Error> {
        ndarray::ArrayView::from_shape(dims, self.as_slice()).map_err(Error::ConstructArrayView)
    }

    pub(crate) fn num_dims(&self) -> usize {
        (unsafe { tflite_sys::TfLiteTensorNumDims(self.tensor as *const _) }) as usize
    }

    pub(crate) fn dim(&self, dim_index: usize) -> usize {
        (unsafe { tflite_sys::TfLiteTensorDim(self.tensor as *const _, dim_index as i32) }) as usize
    }

    pub(crate) fn dims(&self) -> Vec<usize> {
        (0..self.num_dims()).map(|d| self.dim(d)).collect()
    }

    pub(crate) fn byte_size(&self) -> usize {
        (unsafe { tflite_sys::TfLiteTensorByteSize(self.tensor as *const _) }) as usize
    }

    pub(crate) fn copy_from_buffer(&mut self, buf: &[u8]) -> Result<(), Error> {
        let status = unsafe {
            tflite_sys::TfLiteTensorCopyFromBuffer(self.tensor, buf.as_ptr() as _, buf.len())
        };
        Ok(())
    }

    pub(crate) fn copy_to_buffer(&self, buf: &mut [u8]) -> Result<(), Error> {
        let status = unsafe {
            tflite_sys::TfLiteTensorCopyToBuffer(self.tensor, buf.as_mut_ptr() as _, buf.len())
        };
        Ok(())
    }
}

pub(super) struct Delegate {}

impl Delegate {
    pub(super) fn new() -> Result<Self, Error> {
        Ok(Self {})
    }
}

pub(super) struct Interpreter<'a> {
    interpreter: *mut tflite_sys::TfLiteInterpreter,
    _p: PhantomData<&'a ()>,
}

impl<'a> Interpreter<'a> {
    pub(super) fn new<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let model = Model::new(path)?;
        Ok(Self {
            interpreter: std::ptr::null_mut(),
            _p: PhantomData,
        })
    }

    pub(super) fn invoke(&mut self) -> Result<(), Error> {
        let status = unsafe { tflite_sys::TfLiteInterpreterInvoke(self.interpreter) };
        Ok(())
    }

    pub(super) fn get_input_tensor_count(&self) -> usize {
        (unsafe { tflite_sys::TfLiteInterpreterGetInputTensorCount(self.interpreter) }) as usize
    }

    pub(super) fn get_input_tensor(&'a mut self, index: usize) -> Result<Tensor<'a>, Error> {
        let index = i32::try_from(index).map_err(Error::GetFfiIndex)?;
        let pointer =
            unsafe { tflite_sys::TfLiteInterpreterGetInputTensor(self.interpreter, index) };
        if pointer.is_null() {
            return Err(Error::GetInputTensor);
        }
        Tensor::new(pointer)
    }

    pub(super) fn get_output_tensor_count(&self) -> usize {
        (unsafe { tflite_sys::TfLiteInterpreterGetOutputTensorCount(self.interpreter) }) as usize
    }

    pub(super) fn get_output_tensor(&'a mut self, index: usize) -> Result<Tensor<'a>, Error> {
        let index = i32::try_from(index).map_err(Error::GetFfiIndex)?;
        let pointer =
            unsafe { tflite_sys::TfLiteInterpreterGetOutputTensor(self.interpreter, index) };
        if pointer.is_null() {
            return Err(Error::GetOutputTensor);
        }
        Tensor::new(pointer as _)
    }
}

impl<'a> Drop for Interpreter<'a> {
    fn drop(&mut self) {
        // # SAFETY: call accepts null pointers and self.interpreter is guaranteed to be valid,
        unsafe {
            tflite_sys::TfLiteInterpreterDelete(self.interpreter);
        };
    }
}
