use crate::{
    decode::hand_rolled::{keypoint_with_score::KeypointWithScore, point::Point},
    error::Error,
};
use num_traits::ToPrimitive;
use ordered_float::NotNan;

pub(super) struct KeypointPriorityQueue(std::collections::BinaryHeap<KeypointWithScore>);

impl KeypointPriorityQueue {
    pub(super) fn new() -> Self {
        Self(Default::default())
    }

    pub(super) fn push(&mut self, item: KeypointWithScore) {
        self.0.push(item);
    }

    pub(super) fn pop(&mut self) -> Option<KeypointWithScore> {
        self.0.pop()
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn build_keypoint<const NUM_KEYPOINTS: usize>(
        &mut self,
        scores: &[f32],
        short_offsets: &[f32],
        height: usize,
        width: usize,
        score_threshold: f32,
        local_maximum_radius: usize,
    ) -> Result<(), Error> {
        let mut score_index = 0;

        for y in 0..height {
            for x in 0..width {
                let mut offset_index = 2 * score_index;

                for id in 0..NUM_KEYPOINTS {
                    let score = scores[score_index];
                    if score >= score_threshold {
                        let mut local_maximum = true;

                        let y_start = y.saturating_sub(local_maximum_radius);
                        let y_end = (y + local_maximum_radius).min(height);

                        for y_current in y_start..y_end {
                            let x_start = x.saturating_sub(local_maximum_radius);
                            let x_end = (x + local_maximum_radius + 1).min(width);
                            for x_current in x_start..x_end {
                                if scores[y_current * width * NUM_KEYPOINTS
                                    + x_current * NUM_KEYPOINTS
                                    + id]
                                    > score
                                {
                                    local_maximum = false;
                                    break;
                                }
                            }
                            if !local_maximum {
                                break;
                            }
                        }

                        if local_maximum {
                            let dy = short_offsets[offset_index];
                            let dx = short_offsets[offset_index + NUM_KEYPOINTS];
                            let y_refined = (y.to_f32().ok_or(Error::ConvertToF32)? + dy)
                                .clamp(0.0, height.to_f32().ok_or(Error::ConvertToF32)? - 1.0);
                            let x_refined = (x.to_f32().ok_or(Error::ConvertToF32)? + dx)
                                .clamp(0.0, width.to_f32().ok_or(Error::ConvertToF32)? - 1.0);
                            self.push(KeypointWithScore {
                                point: Point::new(x_refined, y_refined)?,
                                id,
                                score: NotNan::new(score)
                                    .map_err(|e| Error::ConstructNotNan(e, score))?,
                            })
                        }
                    }
                    score_index += 1;
                    offset_index += 1;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{KeypointPriorityQueue, KeypointWithScore, Point};
    use ndarray::Array;
    use num_traits::cast::ToPrimitive;
    use ordered_float::NotNan;

    #[test]
    fn build_is_valid_with_threshold() {
        const HEIGHT: usize = 5;
        const WIDTH: usize = 4;
        const NUM_KEYPOINTS: usize = 3;

        let mut scores = Array::zeros((HEIGHT, WIDTH, NUM_KEYPOINTS));
        let p1 = KeypointWithScore {
            point: Point::new(1.0, 2.0).unwrap(),
            id: 1,
            score: NotNan::new(1.0).unwrap(),
        };
        let p2 = KeypointWithScore {
            point: Point::new(3.0, 0.0).unwrap(),
            id: 2,
            score: NotNan::new(1.0).unwrap(),
        };
        scores[(
            p1.point.y().to_usize().unwrap(),
            p1.point.x().to_usize().unwrap(),
            p1.id,
        )] = p1.score.into_inner();
        scores[(
            p2.point.y().to_usize().unwrap(),
            p2.point.x().to_usize().unwrap(),
            p2.id,
        )] = p1.score.into_inner();

        let short_offsets = Array::ones((HEIGHT, WIDTH, NUM_KEYPOINTS, 2));
        let mut queue = KeypointPriorityQueue::new();
        queue
            .build_keypoint::<NUM_KEYPOINTS>(
                scores.as_slice().unwrap(),
                short_offsets.as_slice().unwrap(),
                HEIGHT,
                WIDTH,
                0.5,
                1,
            )
            .unwrap();
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn build_is_valid_scores_correct() {
        const HEIGHT: usize = 5;
        const WIDTH: usize = 4;
        const NUM_KEYPOINTS: usize = 3;

        let mut scores = Array::zeros((HEIGHT, WIDTH, NUM_KEYPOINTS));
        let p1 = KeypointWithScore {
            point: Point::new(2.0, 1.0).unwrap(),
            id: 2,
            score: NotNan::new(1.0).unwrap(),
        };
        let p2 = KeypointWithScore {
            point: Point::new(0.0, 3.0).unwrap(),
            id: 1,
            score: NotNan::new(2.0).unwrap(),
        };
        scores[(
            p1.point.y().to_usize().unwrap(),
            p1.point.x().to_usize().unwrap(),
            p1.id,
        )] = p1.score.into_inner();
        scores[(
            p2.point.y().to_usize().unwrap(),
            p2.point.x().to_usize().unwrap(),
            p2.id,
        )] = p2.score.into_inner();

        let short_offsets = Array::ones((HEIGHT, WIDTH, NUM_KEYPOINTS, 2));
        let expected_keypoint1 = KeypointWithScore {
            point: p1.point + Point::new(1.0, 1.0).unwrap(),
            id: p1.id,
            score: p1.score,
        };
        let expected_keypoint2 = KeypointWithScore {
            point: p2.point + Point::new(1.0, 1.0).unwrap(),
            id: p2.id,
            score: p2.score,
        };
        let mut queue = KeypointPriorityQueue::new();
        queue
            .build_keypoint::<NUM_KEYPOINTS>(
                scores.as_slice().unwrap(),
                short_offsets.as_slice().unwrap(),
                HEIGHT,
                WIDTH,
                0.5,
                1,
            )
            .unwrap();
        let top = queue.pop().unwrap();
        assert_eq!(top.score, expected_keypoint2.score);
        assert_eq!(top.point, expected_keypoint2.point);
        assert_eq!(top.id, expected_keypoint2.id);

        let top = queue.pop().unwrap();
        assert_eq!(top.score, expected_keypoint1.score);
        assert_eq!(top.point, expected_keypoint1.point);
        assert_eq!(top.id, expected_keypoint1.id);
    }
}
