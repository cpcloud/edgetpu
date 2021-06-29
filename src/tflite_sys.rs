#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(dead_code)]
#![allow(deref_nullptr)]
#![allow(unreachable_pub)]

use cxx::{type_id, ExternType};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe impl ExternType for TfLiteTensor {
    type Id = type_id!("TfLiteTensor");
    type Kind = cxx::kind::Opaque;
}
