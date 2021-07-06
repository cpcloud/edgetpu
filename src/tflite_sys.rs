#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(dead_code)]
#![allow(deref_nullptr)]
#![allow(unreachable_pub)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe impl cxx::ExternType for TfLiteTensor {
    type Id = cxx::type_id!("TfLiteTensor");
    type Kind = cxx::kind::Opaque;
}
