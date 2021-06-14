use crate::pose::KeypointKind;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("failed to list devices: got null pointer instead")]
    ListDevices,

    #[error("invalid (null) pointer for device")]
    GetDevice,

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

    #[error("tflite error: {0}")]
    TfLite(&'static str),

    #[error("tflite delegate error: {0}")]
    Delegate(&'static str),

    #[error("tflite application error: {0}")]
    Application(&'static str),

    #[error("failed to construct model from file: C API returned null pointer")]
    GetModelFromFile,

    #[error("failed to create TfLiteInterpreterOptions structure")]
    CreateOptions,

    #[error("failed to create interpreter, got null pointer")]
    CreateInterpreter,

    #[cfg(not(target_arch = "aarch64"))]
    #[error("failed to convert keypoint variant to usize: {0:?}")]
    KeypointVariantToUSize(KeypointKind),

    #[error("failed to create edgetpu delegate: got null pointer")]
    CreateEdgeTpuDelegate,

    #[error("failed to crate posenet decoder delegate")]
    CreatePosenetDecoderDelegate,

    #[error("failed to get edgetpu device: no devices found")]
    GetEdgeTpuDevice,

    #[error("failed to get total number of elements")]
    GetTotalNumberOfElements(#[source] opencv::Error),

    #[error("failed to get number of channels in Mat")]
    GetChannels(#[source] opencv::Error),

    #[error("failed to get Mat data")]
    GetMatData(#[source] opencv::Error),
}
