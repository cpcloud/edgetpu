use crate::{
    error::{check_null, Error},
    tflite_sys,
};

/// A safe wrapper around Tensorflow Lite delegates.
pub(crate) struct Delegate {
    delegate: *const tflite_sys::TfLiteDelegate,
    deleter: Box<dyn FnMut(*mut tflite_sys::TfLiteDelegate)>,
}

impl Delegate {
    pub(crate) fn new<D>(
        delegate: *const tflite_sys::TfLiteDelegate,
        deleter: D,
    ) -> Result<Self, Error>
    where
        D: FnMut(*mut tflite_sys::TfLiteDelegate) + 'static,
    {
        Ok(Self {
            delegate: check_null(delegate).ok_or(Error::ConstructDelegate)?,
            deleter: Box::new(deleter),
        })
    }

    pub(super) fn as_mut_ptr(&self) -> *mut tflite_sys::TfLiteDelegate {
        self.delegate as _
    }
}

impl Drop for Delegate {
    fn drop(&mut self) {
        (self.deleter)(self.delegate as _)
    }
}
