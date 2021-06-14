use crate::{error::Error, tflite_sys};
use std::{ffi::CString, marker::PhantomData};

pub(crate) struct Devices<'a> {
    devices: *mut tflite_sys::edgetpu_device,
    len: usize,
    _p: PhantomData<&'a ()>,
}

pub(crate) struct Device<'a> {
    device: *mut tflite_sys::edgetpu_device,
    _p: PhantomData<&'a ()>,
}

impl<'a> Device<'a> {
    fn new(device: *mut tflite_sys::edgetpu_device) -> Result<Self, Error> {
        if device.is_null() {
            return Err(Error::GetDevice);
        }
        Ok(Self {
            device,
            _p: PhantomData,
        })
    }

    pub(crate) fn r#type(&self) -> tflite_sys::edgetpu_device_type {
        // SAFETY: self.device is guaranteed to be non-null
        unsafe { *self.device }.type_
    }
}

impl<'a> Devices<'a> {
    pub(crate) fn new() -> Result<Self, Error> {
        let mut len: usize = 0;

        // # SAFETY: check for nullness after return; len is guaranteed to point to valid data
        let devices = unsafe { tflite_sys::edgetpu_list_devices(&mut len) };
        if devices.is_null() {
            return Err(Error::ListDevices);
        }

        Ok(Self {
            devices,
            len,
            _p: PhantomData,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn iter(&'a self) -> impl Iterator<Item = Result<Device<'a>, Error>> {
        // SAFETY: self.devices + 0..self.len() is guaranteed to be non-null
        let devices = self.devices;
        (0..self.len()).map(move |offset| Device::new(unsafe { devices.add(offset) }))
    }
}

impl<'a> Drop for Devices<'a> {
    fn drop(&mut self) {
        unsafe {
            tflite_sys::edgetpu_free_devices(self.devices);
        }
    }
}

pub(crate) struct Delegate<'del> {
    pub(crate) delegate: *mut tflite_sys::TfLiteDelegate,
    _p: PhantomData<&'del ()>,
}

impl<K, V> std::convert::TryFrom<(K, V)> for tflite_sys::edgetpu_option
where
    K: Into<String>,
    V: Into<String>,
{
    type Error = Error;

    fn try_from((k, v): (K, V)) -> Result<Self, Self::Error> {
        Ok(Self {
            name: CString::new(k.into())
                .map_err(Error::KeyToCString)?
                .as_ptr(),
            value: CString::new(v.into())
                .map_err(Error::ValueToCString)?
                .as_ptr(),
        })
    }
}

impl<'del> Delegate<'del> {
    pub(super) fn from_device<'dev>(device: Device<'dev>) -> Result<Self, Error> {
        let delegate = unsafe {
            tflite_sys::edgetpu_create_delegate(
                device.r#type(),
                std::ptr::null(),
                std::ptr::null(),
                0,
            )
        };
        Ok(Self {
            delegate,
            _p: PhantomData,
        })
    }
}

impl<'a> Drop for Delegate<'a> {
    fn drop(&mut self) {
        // # SAFETY: self.delegate is guaranteed to be valid
        unsafe {
            tflite_sys::edgetpu_free_delegate(self.delegate);
        }
    }
}
