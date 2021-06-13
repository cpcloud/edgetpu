pub(super) struct Model {
    model: *mut tflite_sys::TfLiteModel,
}

impl Model {
    pub(super) fn new<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        Self {
            model: unsafe { tflite_sys::TfLiteModelCreateFromFile(path.as_ref().as_bytes()) },
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { tflite_sys::TfLiteModelDelete(&mut self.model) };
    }
}

pub(super) struct Tensor {
    tensor: *mut tflite_sys::TfLiteTensor,
}

impl Tensor {
    fn num_dims(&self) -> usize {
        (unsafe { tflite_sys::TfLiteTensorNumDims(self.tensor as *const _) }) as usize
    }

    fn dim(&self, dim_index: usize) -> usize {
        (unsafe { tflite_sys::TfLiteTensorDim(self.tensor as *const _, dim_index as i32) }) as usize
    }

    fn byte_size(&self) -> usize {
        (unsafe { tflite_sys::TfLiteTensorByteSize(self.tensor as *const _, dim_index as i32) })
            as usize
    }

    fn copy_from_buffer(&mut self, buf: &[u8]) -> Result<(), Error> {
        let status = unsafe {
            tflite_sys::TfLiteTensorCopyFromBuffer(
                &mut self.tensor,
                buf.as_ptr() as *const _,
                buf.len(),
            )
        };
        Ok(())
    }

    fn copy_to_buffer(&self, buf: &mut [u8]) -> Result<(), Error> {
        let status = unsafe {
            tflite_sys::TfLiteTensorCopyToBuffer(
                &self.tensor,
                buf.as_mut_ptr() as *mut _,
                buf.len(),
            )
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

pub(super) struct Interpreter {
    interpreter: tflite_sys::TfLiteInterpreter,
}

impl Interpreter {
    pub(super) fn new(model: Model) -> Result<Self, Error> {}

    pub(super) fn invoke(&mut self) -> Result<(), Error> {
        let status = unsafe { tflite_sys::TfLiteInterpreterInvoke(&mut self.interpreter) };
        Ok(())
    }

    pub(super) fn get_input_tensor_count(&self) -> usize {
        let num_tensors =
            unsafe { tflite_sys::TfLiteInterpreterGetInputTensorCount(&self.interpreter) };
        num_tensors as usize
    }

    pub(super) fn get_input_tensor(&mut self, index: usize) -> Option<&mut Tensor> {}

    pub(super) fn inputs(&self) -> Vec<i32> {
        let mut result = vec![];
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        tflite_sys::TfLiteInterpreterDelete(&mut self.interpreter);
    }
}
