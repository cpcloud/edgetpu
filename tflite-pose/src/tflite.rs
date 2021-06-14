use crate::{edgetpu::Devices, error::Error, tflite_sys};
use std::{
    convert::TryFrom, ffi::CString, marker::PhantomData, os::unix::ffi::OsStrExt, path::Path,
};

pub(crate) struct Model<'m> {
    model: *mut tflite_sys::TfLiteModel,
    _p: PhantomData<&'m ()>,
}

fn path_to_c_string<P>(path: P) -> Result<CString, Error>
where
    P: AsRef<Path>,
{
    CString::new(path.as_ref().as_os_str().as_bytes()).map_err(Error::PathToCString)
}

impl<'m> Model<'m> {
    pub(crate) fn new<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let path_ref = path.as_ref();
        let path_bytes = path_to_c_string(path_ref)?;
        // # SAFETY: path_bytes.as_ptr() is guaranteed to be valid
        let model = check_null_mut(
            unsafe { tflite_sys::TfLiteModelCreateFromFile(path_bytes.as_ptr()) },
            || Error::GetModelFromFile,
        )?;

        Ok(Self {
            model,
            _p: PhantomData,
        })
    }

    // SAFETY: You must _not_ deallocate the returned pointer
    fn as_mut_ptr(&mut self) -> *mut tflite_sys::TfLiteModel {
        self.model
    }
}

impl<'m> Drop for Model<'m> {
    fn drop(&mut self) {
        // SAFETY: self.model is guaranteed to be valid
        unsafe {
            tflite_sys::TfLiteModelDelete(self.model);
        }
    }
}

fn status_to_result(status: tflite_sys::TfLiteStatus) -> Result<(), Error> {
    match status {
        tflite_sys::TfLiteStatus::kTfLiteOk => Ok(()),
        tflite_sys::TfLiteStatus::kTfLiteError => Err(Error::TfLite),
        tflite_sys::TfLiteStatus::kTfLiteDelegateError => Err(Error::Delegate),
        tflite_sys::TfLiteStatus::kTfLiteApplicationError => Err(Error::Application),
        _ => panic!("unknown error code: {:?}", status),
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

    pub(crate) fn dim(&self, dim_index: usize) -> usize {
        // # SAFETY: self.tensor is guaranteed to be valid
        usize::try_from(unsafe {
            tflite_sys::TfLiteTensorDim(
                self.tensor as *const _,
                i32::try_from(dim_index).expect("failed to convert dim_index usize to i32"),
            )
        })
        .expect("failed to convert dim i32 to usize")
    }

    pub(crate) fn byte_size(&self) -> usize {
        // # SAFETY: self.tensor is guaranteed to be valid
        usize::try_from(unsafe { tflite_sys::TfLiteTensorByteSize(self.tensor as _) })
            .expect("failed to convert byte_size i32 to usize")
    }

    pub(crate) fn copy_from_buffer(&mut self, buf: &[u8]) -> Result<(), Error> {
        // SAFETY: buf is guaranteed to be valid and of len buf.len()
        status_to_result(unsafe {
            tflite_sys::TfLiteTensorCopyFromBuffer(self.tensor, buf.as_ptr() as _, buf.len())
        })
    }
}

pub(crate) struct Interpreter<'i, 'd, 'm> {
    interpreter: *mut tflite_sys::TfLiteInterpreter,
    options: *mut tflite_sys::TfLiteInterpreterOptions,
    _devices: Devices<'d>,
    _model: Model<'m>,
    _p: PhantomData<&'i ()>,
}

mod coral {
    use super::tflite_sys;

    extern "C" {
        #[link(name = "posenet_decoder")]
        pub fn tflite_plugin_create_delegate(
            options_keys: *mut *mut std::os::raw::c_char,
            options_values: *mut *mut std::os::raw::c_char,
            num_options: usize,
            error_handler: Option<unsafe extern "C" fn(_: *const std::os::raw::c_char)>,
        ) -> *mut tflite_sys::TfLiteDelegate;
    }
}

fn check_null_mut<T>(ptr: *mut T, e: impl FnOnce() -> Error) -> Result<*mut T, Error> {
    if ptr.is_null() {
        Err(e())
    } else {
        Ok(ptr)
    }
}

impl<'i, 'd, 'm> Interpreter<'i, 'd, 'm> {
    pub(crate) fn new<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let (interpreter, options, devices, model) = unsafe {
            // # SAFETY: we check nullness
            let options = check_null_mut(tflite_sys::TfLiteInterpreterOptionsCreate(), || {
                Error::CreateOptions
            })?;

            // add posenet decoder
            let posenet_delegate = check_null_mut(
                coral::tflite_plugin_create_delegate(
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    None,
                ),
                || Error::CreatePosenetDecoderDelegate,
            )?;
            tflite_sys::TfLiteInterpreterOptionsAddDelegate(options, posenet_delegate);

            let devices = Devices::new()?;
            for device in devices.iter() {
                let dev = device?;
                let mut edgetpu_delegate = dev.delegate()?;
                tflite_sys::TfLiteInterpreterOptionsAddDelegate(
                    options,
                    edgetpu_delegate.as_mut_ptr(),
                );
            }

            // SAFETY: all inputs are valid
            tflite_sys::TfLiteInterpreterOptionsSetEnableDelegateFallback(options, false);

            // SAFETY: checkk nullness of interpreter
            let mut model = Model::new(path)?;
            let interpreter = check_null_mut(
                tflite_sys::TfLiteInterpreterCreate(model.as_mut_ptr(), options),
                || Error::CreateInterpreter,
            )?;
            (interpreter, options, devices, model)
        };
        Ok(Self {
            interpreter,
            options,
            _devices: devices,
            _model: model,
            _p: PhantomData,
        })
    }

    pub(crate) fn allocate_tensors(&mut self) -> Result<(), Error> {
        status_to_result(unsafe { tflite_sys::TfLiteInterpreterAllocateTensors(self.interpreter) })
    }

    pub(crate) fn invoke(&mut self) -> Result<(), Error> {
        status_to_result(unsafe { tflite_sys::TfLiteInterpreterInvoke(self.interpreter) })
    }

    pub(crate) fn get_input_tensor(&mut self, index: usize) -> Result<Tensor<'_>, Error> {
        let index = i32::try_from(index).map_err(Error::GetFfiIndex)?;
        let pointer =
            unsafe { tflite_sys::TfLiteInterpreterGetInputTensor(self.interpreter, index) };
        if pointer.is_null() {
            return Err(Error::GetInputTensor);
        }
        Tensor::new(pointer)
    }

    pub(crate) fn get_output_tensor_count(&self) -> usize {
        usize::try_from(unsafe {
            tflite_sys::TfLiteInterpreterGetOutputTensorCount(self.interpreter)
        })
        .expect("failed to convert output tensor count i32 to usize")
    }

    pub(crate) fn get_output_tensor(&self, index: usize) -> Result<Tensor<'_>, Error> {
        let index = i32::try_from(index).map_err(Error::GetFfiIndex)?;
        let pointer =
            unsafe { tflite_sys::TfLiteInterpreterGetOutputTensor(self.interpreter, index) };
        if pointer.is_null() {
            return Err(Error::GetOutputTensor);
        }
        Tensor::new(pointer as _)
    }
}

impl<'i, 'd, 'm> Drop for Interpreter<'i, 'd, 'm> {
    fn drop(&mut self) {
        // # SAFETY: call accepts null pointers and self.interpreter/self.options are guaranteed to
        // be valid.
        unsafe {
            tflite_sys::TfLiteInterpreterDelete(self.interpreter);
            tflite_sys::TfLiteInterpreterOptionsDelete(self.options);
        };
    }
}
