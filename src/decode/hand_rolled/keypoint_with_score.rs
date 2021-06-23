use crate::decode::hand_rolled::point::Point;
use ordered_float::NotNan;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub(super) struct KeypointWithScore {
    pub(super) point: Point,
    pub(super) id: usize,
    pub(super) score: NotNan<f32>,
}

impl Eq for KeypointWithScore {}

impl PartialEq for KeypointWithScore {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

impl Ord for KeypointWithScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}

impl PartialOrd for KeypointWithScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}
