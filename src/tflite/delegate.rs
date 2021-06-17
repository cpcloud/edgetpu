use crate::tflite_sys;

pub(crate) struct Delegate {
    delegate: *const tflite_sys::TfLiteDelegate,
    deleter: Box<dyn FnMut(*mut tflite_sys::TfLiteDelegate)>,
}

impl Delegate {
    pub(crate) fn new<D>(delegate: *const tflite_sys::TfLiteDelegate, deleter: D) -> Self
    where
        D: FnMut(*mut tflite_sys::TfLiteDelegate) + 'static,
    {
        assert!(!delegate.is_null());
        Self {
            delegate,
            deleter: Box::new(deleter),
        }
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
