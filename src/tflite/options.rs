use crate::{
    error::{check_null_mut, Error},
    tflite::Delegate,
    tflite_sys,
};

pub(crate) struct Options {
    options: *mut tflite_sys::TfLiteInterpreterOptions,
}

impl Options {
    pub(super) fn new() -> Result<Self, Error> {
        Ok(Self {
            options: check_null_mut(
                // SAFETY: API is guaranteed to return a valid pointer or null
                unsafe { tflite_sys::TfLiteInterpreterOptionsCreate() },
            )
            .ok_or(Error::CreateOptions)?,
        })
    }

    pub(super) fn add_delegate(&mut self, delegate: &mut Delegate) {
        // SAFETY: self.options and delegate are both valid pointers
        unsafe {
            tflite_sys::TfLiteInterpreterOptionsAddDelegate(self.options, delegate.as_mut_ptr());
        }
    }

    pub(super) fn set_enable_delegate_fallback(&mut self, enable: bool) {
        // SAFETY: self.options is a valid pointer
        unsafe {
            tflite_sys::TfLiteInterpreterOptionsSetEnableDelegateFallback(self.options, enable);
        }
    }

    pub(super) fn as_ptr(&self) -> *const tflite_sys::TfLiteInterpreterOptions {
        self.options as _
    }
}

impl Drop for Options {
    fn drop(&mut self) {
        // SAFETY: self.options is guaranteed to be valid
        unsafe {
            tflite_sys::TfLiteInterpreterOptionsDelete(self.options);
        }
    }
}
