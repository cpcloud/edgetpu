use crate::{
    coral_ffi::ffi,
    edgetpu::Devices,
    error::{check_null_mut, Error},
    tflite::Tensor,
};
use cxx::SharedPtr;
use std::path::{Path, PathBuf};
use tracing::{info, instrument};

#[derive(Clone)]
pub(crate) struct Interpreter {
    interpreter: SharedPtr<ffi::Interpreter>,
    model: SharedPtr<ffi::FlatBufferModel>,
    model_path: PathBuf,
    devices: Devices,
}

impl Interpreter {
    #[instrument(name = "Interpreter::new", skip(path, devices))]
    pub(crate) fn new<P>(path: P, devices: Devices) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        info!(message = "constructing interpreter", path = %path.as_ref().display());
        let edgetpu_context = devices.allocate_one()?;

        let model_path = path.as_ref().to_path_buf();
        let model = unsafe { ffi::make_model(&model_path.display().to_string()) };
        let interpreter =
            unsafe { ffi::make_interpreter_from_model(model.clone(), edgetpu_context) }.unwrap();

        Ok(Self {
            interpreter,
            model_path,
            model,
            devices,
        })
    }

    pub(crate) fn raw(&self) -> SharedPtr<ffi::Interpreter> {
        self.interpreter.clone()
    }

    pub(crate) fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub(crate) fn get_output_tensor_count(&self) -> usize {
        unsafe { ffi::get_output_tensor_count(self.interpreter.clone()) }
    }

    pub(crate) fn get_output_tensor(&self, index: usize) -> Result<Tensor<'_>, Error> {
        Tensor::new(
            check_null_mut(unsafe { ffi::get_output_tensor(self.interpreter.clone(), index) })
                .ok_or(Error::GetOutputTensor)?,
        )
    }

    pub(crate) fn get_output_tensor_by_name(&self, name: &str) -> Result<Tensor<'_>, Error> {
        for index in 0..self.get_output_tensor_count() {
            let tensor = self.get_output_tensor(index)?;
            if tensor.name()? == name {
                return Ok(tensor);
            }
        }
        Err(Error::GetOutputTensorByName(name.to_owned()))
    }
}
