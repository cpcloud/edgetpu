use crate::{
    edgetpu::Devices,
    error::{check_null, check_null_mut, tflite_status_to_result, Error},
    tflite::{Delegate, Model, Options, Tensor},
    tflite_sys,
};
use std::{
    convert::TryFrom,
    path::Path,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
};

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

struct RawInterpreter(AtomicPtr<tflite_sys::TfLiteInterpreter>);

impl Drop for RawInterpreter {
    fn drop(&mut self) {
        // # SAFETY: self.interpreter is guaranteed to be valid.
        unsafe {
            tflite_sys::TfLiteInterpreterDelete(self.0.load(Ordering::SeqCst));
        };
    }
}

#[derive(Clone)]
pub(crate) struct Interpreter {
    interpreter: Arc<RawInterpreter>,
    // These fields are never accessed.
    // They are here to ensure that resources created during interpreter construction
    // live as long as the interpreter.
    _options: Options,
    _devices: Devices,
    _model: Model,
    #[cfg(feature = "posenet_decoder")]
    _posenet_decoder_delegate: Delegate,
    _edgetpu_delegate: Delegate,
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

        let mut devices = Devices::new()?;
        if devices.is_empty() {
            return Err(Error::GetEdgeTpuDevice);
        }

        // add posenet decoder
        #[cfg(feature = "posenet_decoder")]
        let posenet_decoder_delegate = {
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
            posenet_decoder_delegate
        };

        let mut edgetpu_delegate = Delegate::try_from(devices.allocate_one()?)?;

        options.add_delegate(&mut edgetpu_delegate);

        options.set_enable_delegate_fallback(false);

        let mut model = Model::new(path)?;
        let interpreter = check_null_mut(
            // SAFETY: model and options are both valid pointers
            unsafe { tflite_sys::TfLiteInterpreterCreate(model.as_mut_ptr(), options.as_ptr()) },
        )
        .ok_or(Error::CreateInterpreter)?;

        tflite_status_to_result(
            unsafe { tflite_sys::TfLiteInterpreterAllocateTensors(interpreter) },
            "failed to allocate tensors",
        )?;

        Ok(Self {
            interpreter: Arc::new(RawInterpreter(AtomicPtr::new(interpreter))),
            _options: options,
            _devices: devices,
            _model: model,
            #[cfg(feature = "posenet_decoder")]
            _posenet_decoder_delegate: posenet_decoder_delegate,
            _edgetpu_delegate: edgetpu_delegate,
        })
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut tflite_sys::TfLiteInterpreter {
        self.interpreter.0.load(Ordering::SeqCst)
    }

    pub(crate) fn as_ptr(&self) -> *mut tflite_sys::TfLiteInterpreter {
        self.interpreter.0.load(Ordering::SeqCst) as _
    }

    pub(crate) fn get_output_tensor_count(&self) -> Result<usize, Error> {
        usize::try_from(unsafe { tflite_sys::TfLiteInterpreterGetOutputTensorCount(self.as_ptr()) })
            .map_err(Error::GetNumOutputTensors)
    }

    pub(crate) fn get_output_tensor(&self, index: usize) -> Result<Tensor<'_>, Error> {
        let index = i32::try_from(index).map_err(Error::GetFfiIndex)?;
        Tensor::new(
            check_null(unsafe {
                tflite_sys::TfLiteInterpreterGetOutputTensor(self.as_ptr(), index)
            })
            .ok_or(Error::GetOutputTensor)? as _,
        )
    }

    pub(crate) fn get_output_tensor_by_name(&self, name: &str) -> Result<Tensor<'_>, Error> {
        for index in 0..self.get_output_tensor_count()? {
            let tensor = self.get_output_tensor(index)?;
            if tensor.name()? == name {
                return Ok(tensor);
            }
        }
        Err(Error::GetOutputTensorByName(name.to_owned()))
    }
}
