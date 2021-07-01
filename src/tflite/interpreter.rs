use crate::{
    edgetpu::Devices,
    error::{check_null, Error},
    ffi::ffi,
    tflite::Tensor,
};
use cxx::{SharedPtr, UniquePtr};
use std::path::{Path, PathBuf};
use tracing::{info, instrument};

pub(crate) struct Interpreter {
    interpreter: SharedPtr<ffi::Interpreter>,
    model_path: PathBuf,
    _model: UniquePtr<ffi::FlatBufferModel>,
    _devices: Devices,
}

impl Interpreter {
    #[instrument(name = "Interpreter::new", skip(path, devices))]
    pub(crate) fn new<P>(path: P, devices: Devices) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        info!(path = %path.as_ref().display());
        let edgetpu_context = devices.allocate_one()?;

        let model_path = path.as_ref().to_path_buf();
        let path = model_path
            .as_os_str()
            .to_str()
            .ok_or_else(|| Error::GetModelPathAsStr(model_path.clone()))?;
        let model = ffi::make_model(path).map_err(Error::MakeModel)?;
        let interpreter = ffi::make_interpreter_from_model(&*model, edgetpu_context)
            .map_err(Error::MakeInterpreterFromModel)?;

        Ok(Self {
            interpreter,
            model_path,
            _model: model,
            _devices: devices,
        })
    }

    pub(crate) fn as_inner(&self) -> SharedPtr<ffi::Interpreter> {
        self.interpreter.clone()
    }

    pub(crate) fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub(crate) fn get_output_tensor_count(&self) -> Result<usize, Error> {
        ffi::get_output_tensor_count(&*self.interpreter).map_err(Error::GetOutputTensorCount)
    }

    pub(crate) fn get_output_tensor(&self, index: usize) -> Result<Tensor<'_>, Error> {
        Tensor::new(
            check_null(
                ffi::get_output_tensor(&*self.interpreter, index)
                    .map_err(Error::GetOutputTensorFromCxx)?,
            )
            .ok_or(Error::GetOutputTensor)?,
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
