use crate::pose::KeypointKind;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to get typed data from OpenCV Mat")]
    GetTypedData(#[source] opencv::Error),

    #[error("failed to list devices: got null pointer instead")]
    ListDevices,

    #[error("invalid (null) pointer for device")]
    GetDevice,

    #[error("no device at index: {0}")]
    GetDeviceAtIndex(usize),

    #[error("failed to convert Path to CString")]
    PathToCString(#[source] std::ffi::NulError),

    #[error("failed to convert usize to i32")]
    GetFfiIndex(#[source] std::num::TryFromIntError),

    #[error("failed to get input tensor: got null pointer instead")]
    GetInputTensor,

    #[error("failed to get output tensor: got null pointer instead")]
    GetOutputTensor,

    #[error("failed to construct array view from TfLiteTensor")]
    ConstructArrayView(#[source] ndarray::ShapeError),

    #[error("failed to convert usize value to keypoint kind: {0}")]
    ConvertUSizeToKeypointKind(usize),

    #[error("tflite error")]
    TfLite,

    #[error("tflite delegate error")]
    Delegate,

    #[error("tflite application error")]
    Application,

    #[error("failed to construct model from file: C API returned null pointer")]
    GetModelFromFile,

    #[error("failed to convert key into edgetpu_option name")]
    KeyToCString(#[source] std::ffi::NulError),

    #[error("failed to convert value into edgetpu_option value")]
    ValueToCString(#[source] std::ffi::NulError),

    #[error("failed to create TfLiteInterpreterOptions structure")]
    CreateOptions,

    #[error("failed to create interpreter, got null pointer")]
    CreateInterpreter,

    #[error("failed to convert keypoint variant to usize: {0:?}")]
    KeypointVariantToUSize(KeypointKind),
}
