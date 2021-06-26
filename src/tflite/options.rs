use std::sync::{
    atomic::{AtomicPtr, Ordering},
    Arc,
};

use crate::{
    error::{check_null_mut, Error},
    tflite::Delegate,
    tflite_sys,
};

struct RawOptions(AtomicPtr<tflite_sys::TfLiteInterpreterOptions>);

impl Drop for RawOptions {
    fn drop(&mut self) {
        // SAFETY: self.options is guaranteed to be valid
        unsafe {
            tflite_sys::TfLiteInterpreterOptionsDelete(self.0.load(Ordering::SeqCst) as _);
        }
    }
}

#[derive(Clone)]
pub(crate) struct Options {
    options: Arc<RawOptions>,
}

impl Options {
    pub(super) fn new() -> Result<Self, Error> {
        Ok(Self {
            options: Arc::new(RawOptions(AtomicPtr::new(
                check_null_mut(
                    // SAFETY: API is guaranteed to return a valid pointer or null
                    unsafe { tflite_sys::TfLiteInterpreterOptionsCreate() },
                )
                .ok_or(Error::CreateOptions)?,
            ))),
        })
    }

    fn as_mut_ptr(&mut self) -> *mut tflite_sys::TfLiteInterpreterOptions {
        self.options.0.load(Ordering::SeqCst)
    }

    pub(super) fn add_delegate<'a, 'b: 'a>(&'a mut self, delegate: &'b mut Delegate) {
        // SAFETY: self.options and delegate are both valid pointers. `delegate`
        // is owned externally and lives at least as long as self.
        unsafe {
            tflite_sys::TfLiteInterpreterOptionsAddDelegate(
                self.as_mut_ptr(),
                delegate.as_mut_ptr(),
            );
        }
    }

    pub(super) fn set_enable_delegate_fallback(&mut self, enable: bool) {
        // SAFETY: self.options is a valid pointer
        unsafe {
            tflite_sys::TfLiteInterpreterOptionsSetEnableDelegateFallback(
                self.as_mut_ptr(),
                enable,
            );
        }
    }

    pub(super) fn as_ptr(&self) -> *const tflite_sys::TfLiteInterpreterOptions {
        self.options.0.load(Ordering::SeqCst) as _
    }
}
