use crate::{
    edgetpu::Devices,
    error::{check_null, check_null_mut, tflite_status_to_result, Error},
    tflite::{Delegate, Model, Options, Tensor},
    tflite_sys,
};
use std::{convert::TryFrom, path::Path};

#[cfg(feature = "posenet_decoder")]
extern "C" {
    #[link(name = "posenet_decoder")]
    fn tflite_plugin_create_delegate(
        options_keys: *mut *mut std::os::raw::c_char,
        options_values: *mut *mut std::os::raw::c_char,
        num_options: usize,
        error_handler: Option<unsafe extern "C" fn(_: *const std::os::raw::c_char)>,
    ) -> *mut tflite_sys::TfLiteDelegate;

    #[link(name = "posenet_decoder")]
    fn tflite_plugin_destroy_delegate(delegate: *mut tflite_sys::TfLiteDelegate);
}

pub(crate) struct Interpreter {
    interpreter: *mut tflite_sys::TfLiteInterpreter,
    // These fields are never accessed.
    // They are here to ensure that resources created during interpreter construction
    // live as long as the interpreter.
    _options: Options,
    _devices: Devices,
    _model: Model,
    _delegates: Box<[Delegate]>,
}

/// Logging callback for the posenet decoder delegate.
#[cfg(feature = "posenet_decoder")]
unsafe extern "C" fn posenet_decoder_delegate_error_handler(msg: *const std::os::raw::c_char) {
    // SAFETY: `msg` is valid for the lifetime of the call, and doesn't
    // change during that lifetime
    eprintln!("{:?}", std::ffi::CStr::from_ptr(msg));
}

impl Interpreter {
    pub(crate) fn new<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let mut options = Options::new()?;

        let devices = Devices::new()?;
        if devices.is_empty() {
            return Err(Error::GetEdgeTpuDevice);
        }

        let mut delegates = Vec::with_capacity(1 + devices.len());

        // add posenet decoder
        #[cfg(feature = "posenet_decoder")]
        {
            let mut posenet_decoder_delegate = Delegate::new(
                check_null_mut(
                    // SAFETY: inputs are valid
                    unsafe {
                        tflite_plugin_create_delegate(
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            0,
                            Some(posenet_decoder_delegate_error_handler),
                        )
                    },
                )
                .ok_or(Error::CreatePosenetDecoderDelegate)?,
                |delegate| unsafe { tflite_plugin_destroy_delegate(delegate) },
            )?;

            options.add_delegate(&mut posenet_decoder_delegate);

            delegates.push(posenet_decoder_delegate);
        }

        for r#type in devices.types() {
            let mut edgetpu_delegate = Delegate::try_from(r#type?)?;

            options.add_delegate(&mut edgetpu_delegate);
            delegates.push(edgetpu_delegate);
        }

        options.set_enable_delegate_fallback(false);

        let mut model = Model::new(path)?;
        let interpreter = check_null_mut(
            // SAFETY: model and options are both valid pointers
            unsafe { tflite_sys::TfLiteInterpreterCreate(model.as_mut_ptr(), options.as_ptr()) },
        )
        .ok_or(Error::CreateInterpreter)?;

        Ok(Self {
            interpreter,
            _options: options,
            _devices: devices,
            _model: model,
            _delegates: delegates.into_boxed_slice(),
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
        Tensor::new(
            check_null_mut(unsafe {
                tflite_sys::TfLiteInterpreterGetInputTensor(self.interpreter, index)
            })
            .ok_or(Error::GetInputTensor)?,
        )
    }

    pub(crate) fn get_output_tensor_count(&self) -> usize {
        usize::try_from(unsafe {
            tflite_sys::TfLiteInterpreterGetOutputTensorCount(self.interpreter)
        })
        .expect("failed to convert output tensor count i32 to usize")
    }

    pub(crate) fn get_output_tensor(&self, index: usize) -> Result<Tensor<'_>, Error> {
        let index = i32::try_from(index).map_err(Error::GetFfiIndex)?;
        Tensor::new(
            check_null(unsafe {
                tflite_sys::TfLiteInterpreterGetOutputTensor(self.interpreter, index)
            })
            .ok_or(Error::GetOutputTensor)? as _,
        )
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        // # SAFETY: self.interpreter is guaranteed to be valid.
        unsafe {
            tflite_sys::TfLiteInterpreterDelete(self.interpreter);
        };
    }
}
