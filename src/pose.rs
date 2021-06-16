#[cfg(feature = "gui")]
use crate::error::Error;
#[cfg(feature = "gui")]
use num_traits::cast::ToPrimitive;

#[derive(Debug, Copy, Clone, num_derive::FromPrimitive, num_derive::ToPrimitive)]
pub(crate) enum KeypointKind {
    Nose,
    LeftEye,
    RightEye,
    LeftEar,
    RightEar,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
}

#[cfg(feature = "gui")]
impl KeypointKind {
    pub(crate) fn idx(self) -> Result<usize, Error> {
        self.to_usize().ok_or(Error::KeypointVariantToUSize(self))
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct Keypoint {
    pub(crate) kind: Option<KeypointKind>,
    pub(crate) point: opencv::core::Point2f,
    pub(crate) score: f32,
}

pub(crate) const NUM_KEYPOINTS: usize = std::mem::variant_count::<KeypointKind>();
pub(crate) type Keypoints = [Keypoint; NUM_KEYPOINTS];

#[derive(Debug, Copy, Clone)]
pub(crate) struct Pose {
    pub(crate) keypoints: Keypoints,
    pub(crate) score: f32,
}

#[cfg(feature = "gui")]
pub(crate) const KEYPOINT_EDGES: [(KeypointKind, KeypointKind); 19] = [
    (KeypointKind::Nose, KeypointKind::LeftEye),
    (KeypointKind::Nose, KeypointKind::RightEye),
    (KeypointKind::Nose, KeypointKind::LeftEar),
    (KeypointKind::Nose, KeypointKind::RightEar),
    (KeypointKind::LeftEar, KeypointKind::LeftEye),
    (KeypointKind::RightEar, KeypointKind::RightEye),
    (KeypointKind::LeftEye, KeypointKind::RightEye),
    (KeypointKind::LeftShoulder, KeypointKind::RightShoulder),
    (KeypointKind::LeftShoulder, KeypointKind::LeftElbow),
    (KeypointKind::LeftShoulder, KeypointKind::LeftHip),
    (KeypointKind::RightShoulder, KeypointKind::RightElbow),
    (KeypointKind::RightShoulder, KeypointKind::RightHip),
    (KeypointKind::LeftElbow, KeypointKind::LeftWrist),
    (KeypointKind::RightElbow, KeypointKind::RightWrist),
    (KeypointKind::LeftHip, KeypointKind::RightHip),
    (KeypointKind::LeftHip, KeypointKind::LeftKnee),
    (KeypointKind::RightHip, KeypointKind::RightKnee),
    (KeypointKind::LeftKnee, KeypointKind::LeftAnkle),
    (KeypointKind::RightKnee, KeypointKind::RightAnkle),
];
