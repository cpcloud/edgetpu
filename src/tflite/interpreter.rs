use crate::{
    edgetpu::Devices,
    error::{check_null, check_null_mut, tflite_status_to_result, Error},
    tflite::{Delegate, Model, Tensor},
    tflite_sys,
};
use std::{convert::TryFrom, path::Path};

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
    options: *mut tflite_sys::TfLiteInterpreterOptions,
    // these fields are never accessed, they are only here to ensure that
    // resources created during interpreter construction live as long as the
    // interpreter
    _devices: Devices,
    _model: Model,
    _delegates: Vec<Delegate>,
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

        let devices = Devices::new()?;
        if devices.is_empty() {
            return Err(Error::GetEdgeTpuDevice);
        }

        let mut delegates = Vec::with_capacity(1 + devices.len());

        // add posenet decoder
        let posenet_decoder_delegate = Delegate::new(
            check_null_mut(
                // SAFETY: the delegate is guaranteed to be valid
                unsafe {
                    tflite_plugin_create_delegate(
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0,
                        None,
                    )
                },
                || Error::CreatePosenetDecoderDelegate,
            )?,
            |delegate| unsafe { tflite_plugin_destroy_delegate(delegate) },
        );

        // SAFETY: options and posenet_delegate are both valid pointers
        unsafe {
            tflite_sys::TfLiteInterpreterOptionsAddDelegate(
                options,
                posenet_decoder_delegate.as_mut_ptr(),
            );
        }

        delegates.push(posenet_decoder_delegate);

        for device in devices.iter() {
            let edgetpu_delegate = device.delegate()?;

            // SAFETY: options and edgetpu_delegate are both valid pointers
            unsafe {
                tflite_sys::TfLiteInterpreterOptionsAddDelegate(
                    options,
                    edgetpu_delegate.as_mut_ptr(),
                );
            }

            delegates.push(edgetpu_delegate);
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
            _delegates: delegates,
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
        // # SAFETY: call accepts null pointers and self.interpreter/self.options are guaranteed to
        // be valid.
        unsafe {
            tflite_sys::TfLiteInterpreterDelete(self.interpreter);
            tflite_sys::TfLiteInterpreterOptionsDelete(self.options);
        };
    }
}
