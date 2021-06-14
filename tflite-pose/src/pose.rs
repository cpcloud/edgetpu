#[derive(Debug, Copy, Clone, num_derive::FromPrimitive)]
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

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct Keypoint {
    pub(crate) kind: Option<KeypointKind>,
    pub(crate) point: opencv::core::Point2f,
    pub(crate) score: f32,
}

pub(crate) type Keypoints = [Keypoint; std::mem::variant_count::<KeypointKind>()];

pub(crate) struct Pose {
    pub(crate) keypoints: Keypoints,
    pub(crate) score: f32,
}

impl Pose {
    pub(crate) fn new(keypoints: Keypoints, score: f32) -> Self {
        Self { keypoints, score }
    }
}

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
