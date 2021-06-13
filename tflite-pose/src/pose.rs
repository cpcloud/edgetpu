use opencv::core::Point2f;

#[derive(Debug, Copy, Clone)]
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

#[derive(Debug, Copy, Clone)]
pub(crate) struct Keypoint {
    kind: KeypointKind,
    point: Point2f,
    score: f64,
}

impl Default for Keypoint {
    fn default() -> Self {
        Self {
            kind: KeypointKind::Nose,
            point: Point2f::new(0.0_f32, 0.0_f32),
            score: 0.0,
        }
    }
}

pub(crate) struct Pose {
    keypoints: [Keypoint; std::mem::variant_count::<KeypointKind>()],
    score: f64,
}

const EDGES: [(KeypointKind, KeypointKind); 19] = [
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
