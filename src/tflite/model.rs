use crate::{
    error::{check_null_mut, Error},
    tflite_sys,
};
use std::{
    ffi::CString,
    os::unix::ffi::OsStrExt,
    path::Path,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
};

struct RawModel(AtomicPtr<tflite_sys::TfLiteModel>);

impl Drop for RawModel {
    fn drop(&mut self) {
        // SAFETY: self.model is guaranteed to be valid
        unsafe {
            tflite_sys::TfLiteModelDelete(self.0.load(Ordering::SeqCst));
        }
    }
}

/// A TFLiteModel wrapper.
#[derive(Clone)]
pub(crate) struct Model {
    /// SAFETY: `model` is owned and not mutated by any other APIs here
    /// or in TFLite.
    model: Arc<RawModel>,
}

fn path_to_c_string<P>(path: P) -> Result<CString, Error>
where
    P: AsRef<Path>,
{
    CString::new(path.as_ref().as_os_str().as_bytes()).map_err(Error::PathToCString)
}

impl Model {
    pub(super) fn new<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let path_ref = path.as_ref();
        let path_bytes = path_to_c_string(path_ref)?;
        Ok(Self {
            model: Arc::new(RawModel(AtomicPtr::new(
                // # SAFETY: path_bytes.as_ptr() is guaranteed to be valid
                check_null_mut(unsafe {
                    tflite_sys::TfLiteModelCreateFromFile(path_bytes.as_ptr())
                })
                .ok_or(Error::GetModelFromFile)?,
            ))),
        })
    }

    // SAFETY: You must _not_ deallocate the returned pointer
    pub(super) fn as_mut_ptr(&mut self) -> *mut tflite_sys::TfLiteModel {
        self.model.0.load(Ordering::SeqCst)
    }
}
