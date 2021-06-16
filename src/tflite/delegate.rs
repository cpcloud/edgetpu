use std::convert::TryFrom;

use crate::{
    error::{check_null, check_null_mut, Error},
    tflite_sys,
};

/// A safe wrapper around Tensorflow Lite delegates.
pub(crate) struct Delegate {
    delegate: *const tflite_sys::TfLiteDelegate,
    deleter: Box<dyn FnMut(*mut tflite_sys::TfLiteDelegate)>,
}

impl TryFrom<tflite_sys::edgetpu_device_type> for Delegate {
    type Error = Error;

    fn try_from(r#type: tflite_sys::edgetpu_device_type) -> Result<Self, Self::Error> {
        Self::new(
            check_null_mut(
                // SAFETY: inputs are all valid, and the return value is checked for null
                unsafe {
                    tflite_sys::edgetpu_create_delegate(
                        r#type,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                    )
                },
            )
            .ok_or(Error::CreateEdgeTpuDelegate)?,
            |delegate| unsafe { tflite_sys::edgetpu_free_delegate(delegate) },
        )
    }
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
