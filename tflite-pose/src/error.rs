#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("failed to convert inference milliseconds to duration")]
    ConvertInferenceMillis(#[source] std::num::TryFromIntError),

    #[error("failed to get typed data from OpenCV Mat")]
    GetTypedData(#[source] opencv::Error),

    #[error("failed to list devices: got null pointer instead")]
    ListDevices,

    #[error("invalid (null) pointer for device")]
    GetDevice,

    #[error("failed to convert Path to CString")]
    PathToCString(#[source] std::ffi::NulError),

    #[error("failed to convert Path to &str")]
    PathToStr,

    #[error("failed to convert usize to i32")]
    GetFfiIndex(#[source] std::num::TryFromIntError),

    #[error("failed to get input tensor: got null pointer instead")]
    GetInputTensor,

    #[error("failed to get output tensor: got null pointer instead")]
    GetOutputTensor,

    #[error("failed to construct array view from TfLiteTensor")]
    ConstructArrayView(#[source] ndarray::Error),

    #[error("failed to convert usize value to keypoint kind: {0}")]
    ConvertUSizeToKeypointKind(usize),
}
