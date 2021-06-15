use crate::{
    edgetpu::Devices,
    error::{check_null, check_null_mut, tflite_status_to_result, Error},
    tflite_sys,
};
use std::{
    convert::TryFrom, ffi::CString, marker::PhantomData, os::unix::ffi::OsStrExt, path::Path,
};

/// A safe TFLiteModel wrapper.
pub(crate) struct Model {
    /// SAFETY: `model` is fully owned.
    model: *mut tflite_sys::TfLiteModel,
}

fn path_to_c_string<P>(path: P) -> Result<CString, Error>
where
    P: AsRef<Path>,
{
    CString::new(path.as_ref().as_os_str().as_bytes()).map_err(Error::PathToCString)
}

impl Model {
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

        Ok(Self { model })
    }

    // SAFETY: You must _not_ deallocate the returned pointer
    fn as_mut_ptr(&mut self) -> *mut tflite_sys::TfLiteModel {
        self.model
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: self.model is guaranteed to be valid
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

    pub(crate) fn dim(&self, dim_index: usize) -> usize {
        // # SAFETY: self.tensor is guaranteed to be valid
        usize::try_from(unsafe {
            tflite_sys::TfLiteTensorDim(
                self.tensor as _,
                i32::try_from(dim_index).expect("failed to convert dim_index usize to i32"),
            )
        })
        .expect("failed to convert dim i32 to usize")
    }

    pub(crate) fn byte_size(&self) -> usize {
        // # SAFETY: self.tensor is guaranteed to be valid
        unsafe { tflite_sys::TfLiteTensorByteSize(self.tensor as _) }
    }

    pub(crate) fn copy_from_buffer(&mut self, buf: &[u8]) -> Result<(), Error> {
        // SAFETY: buf is guaranteed to be valid and of len buf.len()
        tflite_status_to_result(
            unsafe {
                tflite_sys::TfLiteTensorCopyFromBuffer(self.tensor, buf.as_ptr() as _, buf.len())
            },
            "failed to copy from input buffer",
        )
    }
}

type Delegate = (
    *mut tflite_sys::TfLiteDelegate,
    Box<dyn FnOnce(*mut tflite_sys::TfLiteDelegate)>,
);

pub(crate) struct Interpreter {
    interpreter: *mut tflite_sys::TfLiteInterpreter,
    options: *mut tflite_sys::TfLiteInterpreterOptions,
    // these fields are never accessed, they are only here to ensure that
    // resources created by interpreter/options live as long as interpreter/options
    _devices: Devices,
    _model: Model,
    delegates: Vec<Delegate>,
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

        #[link(name = "posenet_decoder")]
        pub fn tflite_plugin_destroy_delegate(delegate: *mut tflite_sys::TfLiteDelegate);
    }
}

impl Interpreter {
    pub(crate) fn new<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let options = check_null_mut(
            // SAFETY: we check nullness, API is guaranteed to return a valid pointer
            unsafe { tflite_sys::TfLiteInterpreterOptionsCreate() },
            || Error::CreateOptions,
        )?;

        // add posenet decoder
        let posenet_decoder_delegate = check_null_mut(
            // SAFETY: the delegate is guaranteed to be valid
            unsafe {
                coral::tflite_plugin_create_delegate(
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    None,
                )
            },
            || Error::CreatePosenetDecoderDelegate,
        )?;

        // SAFETY: options and posenet_delegate are both valid pointers
        unsafe {
            tflite_sys::TfLiteInterpreterOptionsAddDelegate(options, posenet_decoder_delegate);
        }

        let devices = Devices::new()?;
        if devices.is_empty() {
            return Err(Error::GetEdgeTpuDevice);
        }

        let mut delegates: Vec<(_, Box<dyn FnOnce(_)>)> = Vec::with_capacity(devices.len());
        delegates.push((
            posenet_decoder_delegate,
            // SAFETY: delegate is a valid pointer
            Box::new(|delegate| unsafe { coral::tflite_plugin_destroy_delegate(delegate) }),
        ));

        for device in devices.iter() {
            let edgetpu_delegate = device.delegate()?;

            // SAFETY: options and edgetpu_delegate are both valid pointers
            unsafe {
                tflite_sys::TfLiteInterpreterOptionsAddDelegate(options, edgetpu_delegate);
            }

            delegates.push((
                edgetpu_delegate,
                Box::new(|delegate| unsafe { tflite_sys::edgetpu_free_delegate(delegate) }),
            ));
        }

        // SAFETY: options is a valid pointer
        unsafe {
            tflite_sys::TfLiteInterpreterOptionsSetEnableDelegateFallback(options, false);
        }

        let mut model = Model::new(path)?;
        let interpreter = check_null_mut(
            // SAFETY: model and options are both valid pointers
            unsafe { tflite_sys::TfLiteInterpreterCreate(model.as_mut_ptr(), options) },
            || Error::CreateInterpreter,
        )?;

        Ok(Self {
            interpreter,
            options,
            _devices: devices,
            _model: model,
            delegates,
        })
    }

    pub(crate) fn allocate_tensors(&mut self) -> Result<(), Error> {
        tflite_status_to_result(
            unsafe { tflite_sys::TfLiteInterpreterAllocateTensors(self.interpreter) },
            "failed to allocate tensors",
        )
    }

    pub(crate) fn invoke(&mut self) -> Result<(), Error> {
        tflite_status_to_result(
            unsafe { tflite_sys::TfLiteInterpreterInvoke(self.interpreter) },
            "model invocation failed",
        )
    }

    pub(crate) fn get_input_tensor(&mut self, index: usize) -> Result<Tensor<'_>, Error> {
        let index = i32::try_from(index).map_err(Error::GetFfiIndex)?;
        let pointer = check_null_mut(
            unsafe { tflite_sys::TfLiteInterpreterGetInputTensor(self.interpreter, index) },
            || Error::GetInputTensor,
        )?;
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
        let pointer = check_null(
            unsafe { tflite_sys::TfLiteInterpreterGetOutputTensor(self.interpreter, index) },
            || Error::GetOutputTensor,
        )?;
        Tensor::new(pointer as _)
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        for (delegate, dropper) in self.delegates.drain(..) {
            dropper(delegate);
        }

        // # SAFETY: call accepts null pointers and self.interpreter/self.options are guaranteed to
        // be valid.
        unsafe {
            tflite_sys::TfLiteInterpreterDelete(self.interpreter);
            tflite_sys::TfLiteInterpreterOptionsDelete(self.options);
        };
    }
}
