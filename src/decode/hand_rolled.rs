use crate::{
    error::Error,
    pose::{self, NUM_KEYPOINTS},
    tflite,
};
use ndarray::{Array, Array1, Array2, Array3};
use num_traits::{cast::ToPrimitive, Zero};
use opencv::core::Point2f;
use ordered_float::NotNan;
use std::{cmp::Ordering, collections::BinaryHeap};

#[derive(Debug, Clone, Copy, structopt::StructOpt)]
pub(crate) struct Decoder {
    #[structopt(short, long, default_value = "16")]
    pub(crate) output_stride: u8,
    #[structopt(short = "-M", long, default_value = "100")]
    pub(crate) max_pose_detections: usize,
    #[structopt(short, long, default_value = "0.5")]
    pub(crate) score_threshold: f32,
    #[structopt(short, long, default_value = "20")]
    pub(crate) nms_radius: usize,
    #[structopt(short = "-r", long, default_value = "5")]
    pub(crate) mid_short_offset_refinement_steps: usize,
}

type PoseKeypoints<const N: usize> = [Point2f; N];
type PoseKeypointScores<const N: usize> = [f32; N];

impl Default for Decoder {
    fn default() -> Self {
        Self {
            output_stride: 16,
            max_pose_detections: 100,
            score_threshold: 0.5,
            nms_radius: 20,
            mid_short_offset_refinement_steps: 5,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct KeypointWithScore {
    point: Point2f,
    id: usize,
    score: ordered_float::NotNan<f32>,
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

struct DecreasingScoreKeypointPriorityQueue(BinaryHeap<KeypointWithScore>);

impl std::ops::Deref for DecreasingScoreKeypointPriorityQueue {
    type Target = BinaryHeap<KeypointWithScore>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for DecreasingScoreKeypointPriorityQueue {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl DecreasingScoreKeypointPriorityQueue {
    fn new() -> Self {
        Self(BinaryHeap::new())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_keypoint(
        &mut self,
        scores: &[f32],
        short_offsets: &[f32],
        height: usize,
        width: usize,
        num_keypoints: usize,
        score_threshold: f32,
        local_maximum_radius: usize,
    ) -> Result<(), Error> {
        let mut score_index = 0;

        for y in 0..height {
            for x in 0..width {
                let mut offset_index = 2 * score_index;

                for j in 0..num_keypoints {
                    let score = scores[score_index];
                    if score >= score_threshold {
                        let mut local_maximum = true;

                        let y_start = y.saturating_sub(local_maximum_radius);
                        let y_end = (y + local_maximum_radius).min(height);

                        for y_current in y_start..y_end {
                            let x_start = x.saturating_sub(local_maximum_radius);
                            let x_end = (x + local_maximum_radius + 1).min(width);
                            for x_current in x_start..x_end {
                                if scores[y_current * width * num_keypoints
                                    + x_current * num_keypoints
                                    + j]
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
                            let dx = short_offsets[offset_index + num_keypoints];
                            let y_refined = (y.to_f32().ok_or(Error::ConvertToF32)? + dy)
                                .clamp(0.0, height.to_f32().ok_or(Error::ConvertToF32)? - 1.0);
                            let x_refined = (x.to_f32().ok_or(Error::ConvertToF32)? + dx)
                                .clamp(0.0, width.to_f32().ok_or(Error::ConvertToF32)? - 1.0);
                            self.0.push(KeypointWithScore {
                                point: Point2f::new(x_refined, y_refined),
                                id: j,
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

const NUM_EDGES: usize = pose::constants::EDGE_LIST.len();

#[derive(Debug, PartialEq, PartialOrd, Eq, Ord)]
struct AdjacencyList {
    child_ids: Vec<Vec<usize>>,
    edge_ids: Vec<Vec<usize>>,
}

impl AdjacencyList {
    fn new(n: usize) -> Self {
        Self {
            child_ids: vec![vec![]; n],
            edge_ids: vec![vec![]; n],
        }
    }
}

impl Default for AdjacencyList {
    fn default() -> Self {
        Self {
            child_ids: Default::default(),
            edge_ids: Default::default(),
        }
    }
}

fn build_adjacency_list() -> AdjacencyList {
    let mut adjacency_list = AdjacencyList::new(NUM_KEYPOINTS);
    for (k, (parent, child)) in pose::constants::EDGE_LIST.iter().enumerate() {
        let parent_id = parent.to_usize().unwrap();
        let child_id = child.to_usize().unwrap();
        adjacency_list.child_ids[parent_id].push(child_id);
        adjacency_list.edge_ids[parent_id].push(k);
    }
    adjacency_list
}

fn decreasing_arg_sort(scores: &[f32], indices: &mut [usize]) {
    indices.iter_mut().enumerate().for_each(|(src, dst)| {
        *dst = src;
    });
    indices.sort_by_key(|&i| std::cmp::Reverse(NotNan::new(scores[i]).unwrap()))
}

impl Decoder {
    #[allow(clippy::type_complexity)]
    fn decode_all_poses<const N: usize>(
        &self,
        scores: &[f32],
        short_offsets: &[f32],
        mid_offsets: &[f32],
        height: usize,
        width: usize,
    ) -> Result<(Array1<f32>, Array2<f32>, Array3<f32>), Error> {
        let &Self {
            max_pose_detections,
            score_threshold,
            output_stride,
            ..
        } = self;
        let output_stride = f32::from(output_stride);
        let nms_radius =
            self.nms_radius.to_f32().ok_or(Error::ConvertToF32)? * output_stride.recip();
        let mut pose_scores = vec![0.0; max_pose_detections];
        let mut pose_keypoint_scores = vec![[0.0; N]; max_pose_detections];
        let mut pose_keypoints = vec![[Point2f::default(); N]; max_pose_detections];

        let local_maximum_radius = pose::constants::LOCAL_MAXIMUM_RADIUS;
        let min_score_logit = log_odds(score_threshold);

        let mut queue = DecreasingScoreKeypointPriorityQueue::new();
        queue.build_keypoint(
            scores,
            short_offsets,
            height,
            width,
            pose::NUM_KEYPOINTS,
            min_score_logit,
            local_maximum_radius,
        )?;

        let adjacency_list = build_adjacency_list();
        let topk = pose::NUM_KEYPOINTS;
        let mut indices = [0_usize; pose::NUM_KEYPOINTS];
        let mut pose_counter = 0usize;

        let mut all_instance_scores = Vec::with_capacity(max_pose_detections);
        let mut scratch_poses = vec![[Point2f::default(); N]; max_pose_detections];
        let mut scratch_keypoint_scores = vec![[0.0; N]; max_pose_detections];

        while let Some(root) = queue.0.pop() {
            if pose_counter >= max_pose_detections {
                break;
            }

            if !pass_keypoint_nms(&pose_keypoints[..pose_counter], root, nms_radius.powi(2)) {
                continue;
            }

            let next_pose = &mut scratch_poses[pose_counter];
            next_pose.fill(Point2f::new(-1.0, -1.0));

            let next_scores = &mut scratch_keypoint_scores[pose_counter];
            next_scores.fill(-1e5);

            backtrack_decode_pose(
                scores,
                short_offsets,
                mid_offsets,
                height,
                width,
                pose::NUM_KEYPOINTS,
                NUM_EDGES,
                &root,
                &adjacency_list,
                self.mid_short_offset_refinement_steps,
                next_pose,
                next_scores,
            )?;
            next_scores.iter_mut().for_each(|v| *v = sigmoid(*v));
            decreasing_arg_sort(next_scores, &mut indices);
            let mut instance_score = 0.0_f32;
            for j in 0..topk {
                instance_score += next_scores[indices[j]];
            }
            instance_score /= topk.to_f32().ok_or(Error::ConvertToF32)?;
            if instance_score >= score_threshold {
                pose_counter += 1;
                all_instance_scores.push(instance_score);
            }
        }

        let mut decreasing_indices = vec![0; all_instance_scores.len()];
        decreasing_arg_sort(&all_instance_scores, &mut decreasing_indices);

        perform_soft_keypoint_nms(
            &decreasing_indices,
            &scratch_poses,
            &scratch_keypoint_scores,
            pose::NUM_KEYPOINTS,
            nms_radius.powi(2),
            topk,
            &mut all_instance_scores,
        );

        decreasing_arg_sort(&all_instance_scores, &mut decreasing_indices);

        pose_counter = 0;

        for index in decreasing_indices
            .into_iter()
            .take_while(|&index| all_instance_scores[index] < score_threshold)
        {
            for k in 0..pose::NUM_KEYPOINTS {
                pose_keypoints[pose_counter][k].y = scratch_poses[index][k].y * output_stride;
                pose_keypoints[pose_counter][k].x = scratch_poses[index][k].x * output_stride;
            }

            pose_keypoint_scores[pose_counter] = scratch_keypoint_scores[index];
            pose_scores[pose_counter] = all_instance_scores[index];
            pose_counter += 1;
        }

        let mut pose_keypoint_scores_arr = Array::zeros((max_pose_detections, pose::NUM_KEYPOINTS));
        let mut pose_keypoints_arr = Array::zeros((max_pose_detections, pose::NUM_KEYPOINTS, 2));

        for (i, (keypoints, keypoint_scores)) in pose_keypoints
            .into_iter()
            .zip(pose_keypoint_scores)
            .enumerate()
        {
            for (j, (&Point2f { x, y }, score)) in keypoints.iter().zip(keypoint_scores).enumerate()
            {
                pose_keypoint_scores_arr[(i, j)] = score;
                pose_keypoints_arr[(i, j, 0)] = x;
                pose_keypoints_arr[(i, j, 1)] = y;
            }
        }
        Ok((
            pose_scores.into(),
            pose_keypoint_scores_arr,
            pose_keypoints_arr,
        ))
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn log_odds(x: f32) -> f32 {
    -(1.0 / (x + 1e-6) - 1.0).ln()
}

fn compute_squared_distance(a: Point2f, b: Point2f) -> f32 {
    let delta = b - a;
    delta.dot(delta)
}

fn build_linear_interpolation(x: f32, n: usize) -> (usize, usize, f32) {
    let x_proj = x.clamp(0.0, n.to_f32().unwrap() - 1.0);
    let floor_f = x_proj.floor();
    let ceil_f = x_proj.ceil();
    (
        floor_f.to_usize().unwrap(),
        ceil_f.to_usize().unwrap(),
        x - floor_f,
    )
}

fn build_bilinear_interpolation(
    y: f32,
    x: f32,
    height: usize,
    width: usize,
    num_channels: usize,
) -> (usize, usize, usize, usize, f32, f32) {
    let (y_floor, y_ceil, y_lerp) = build_linear_interpolation(y, height);
    let (x_floor, x_ceil, x_lerp) = build_linear_interpolation(x, width);
    (
        (y_floor * width + x_floor) * num_channels,
        (y_floor * width + x_ceil) * num_channels,
        (y_ceil * width + x_floor) * num_channels,
        (y_ceil * width + x_ceil) * num_channels,
        y_lerp,
        x_lerp,
    )
}

#[allow(clippy::too_many_arguments)]
fn sample_tensor_at_multiple_channels(
    tensor: &[f32],
    height: usize,
    width: usize,
    num_channels: usize,
    y: f32,
    x: f32,
    result_channels: &[usize],
    result: &mut [NotNan<f32>],
) -> Result<(), Error> {
    let (top_left, top_right, bottom_left, bottom_right, y_lerp, x_lerp) =
        build_bilinear_interpolation(y, x, height, width, num_channels);
    assert_eq!(result_channels.len(), result.len());
    result
        .iter_mut()
        .zip(result_channels)
        .try_for_each(|(dst, &c)| {
            let value = (1.0 - y_lerp)
                * ((1.0 - x_lerp) * tensor[top_left + c] + x_lerp * tensor[top_right + c])
                + y_lerp
                    * ((1.0 - x_lerp) * tensor[bottom_left + c]
                        + x_lerp * tensor[bottom_right + c]);
            *dst = NotNan::new(value).map_err(|e| Error::ConstructNotNan(e, value))?;
            Ok(())
        })
}

fn sample_tensor_at_single_channel(
    tensor: &[f32],
    height: usize,
    width: usize,
    num_channels: usize,
    point: Point2f,
    c: usize,
) -> Result<NotNan<f32>, Error> {
    let mut result = [NotNan::zero(); 1];
    let c = [c];
    sample_tensor_at_multiple_channels(
        tensor,
        height,
        width,
        num_channels,
        point.y,
        point.x,
        &c,
        &mut result,
    )?;
    let [value] = result;
    Ok(value)
}

#[allow(clippy::too_many_arguments)]
fn find_displaced_position(
    short_offsets: &[f32],
    mid_offsets: &[f32],
    height: usize,
    width: usize,
    num_keypoints: usize,
    num_edges: usize,
    source: Point2f,
    edge_id: usize,
    target_id: usize,
    mid_short_offset_refinement_steps: usize,
) -> Result<Point2f, Error> {
    let mut y = source.y;
    let mut x = source.x;
    let mut offsets = [NotNan::zero(); 2];
    // Follow the mid-range offsets.
    let mut channels = [edge_id, num_edges + edge_id];
    // Total size of mid_offsets is height x width x 2*2*num_edges
    sample_tensor_at_multiple_channels(
        mid_offsets,
        height,
        width,
        2 * 2 * num_edges,
        y,
        x,
        &channels,
        &mut offsets,
    )?;
    y = (y + offsets[0].into_inner()).clamp(0.0, height.to_f32().unwrap() - 1.0);
    x = (x + offsets[1].into_inner()).clamp(0.0, width.to_f32().unwrap() - 1.0);
    // Refine by the short-range offsets.
    channels[0] = target_id;
    channels[1] = num_keypoints + target_id;

    for _ in 0..mid_short_offset_refinement_steps {
        sample_tensor_at_multiple_channels(
            short_offsets,
            height,
            width,
            2 * num_keypoints,
            y,
            x,
            &channels,
            &mut offsets,
        )?;
        y = (y + offsets[0].into_inner()).clamp(0.0, height.to_f32().unwrap() - 1.0);
        x = (x + offsets[1].into_inner()).clamp(0.0, width.to_f32().unwrap() - 1.0);
    }

    Ok(Point2f::new(x, y))
}

#[allow(clippy::too_many_arguments)]
fn backtrack_decode_pose<const N: usize>(
    scores: &[f32],
    short_offsets: &[f32],
    mid_offsets: &[f32],
    height: usize,
    width: usize,
    num_keypoints: usize,
    num_edges: usize,
    root: &KeypointWithScore,
    adjacency_list: &AdjacencyList,
    mid_short_offset_refinement_steps: usize,
    pose_keypoints: &mut PoseKeypoints<N>,
    keypoint_scores: &mut PoseKeypointScores<N>,
) -> Result<(), Error> {
    let root_score =
        sample_tensor_at_single_channel(scores, height, width, num_keypoints, root.point, root.id)?;

    // Used in order to put candidate keypoints in a priority queue w.r.t. their
    // score. Keypoints with higher score have higher priority and will be
    // decoded/processed first.
    let mut decode_queue = DecreasingScoreKeypointPriorityQueue::new();
    decode_queue.push(KeypointWithScore {
        point: root.point,
        id: root.id,
        score: root_score,
    });

    // Keeps track of the keypoints whose position has already been decoded.
    let mut keypoint_decoded = vec![false; num_keypoints];

    // The top element in the queue is the next keypoint to be processed.
    while let Some(KeypointWithScore { point, id, score }) = decode_queue.pop() {
        if keypoint_decoded[id] {
            continue;
        }

        pose_keypoints[id] = point;
        keypoint_scores[id] = score.into_inner();
        keypoint_decoded[id] = true;

        // Add the children of the current keypoint that have not been decoded yet
        // to the priority queue.
        let num_children = adjacency_list.child_ids[id].len();
        for j in 0..num_children {
            let child_id = adjacency_list.child_ids[id][j];
            let mut edge_id = adjacency_list.edge_ids[id][j];
            if keypoint_decoded[child_id] {
                continue;
            }

            // The mid-offsets block is organized as 4 blocks of kNumEdges:
            // [fwd Y offsets][fwd X offsets][bwd Y offsets][bwd X offsets]
            // OTOH edge_id is [0,kNumEdges) for forward edges and
            // [kNumEdges, 2*kNumEdges) for backward edges.
            // Thus if the edge is a backward edge (>kNumEdges) then we need
            // to start 16 indices later to be correctly aligned with the mid-offsets.
            if edge_id > NUM_EDGES {
                edge_id += NUM_EDGES;
            }

            let child_point = find_displaced_position(
                short_offsets,
                mid_offsets,
                height,
                width,
                num_keypoints,
                num_edges,
                point,
                edge_id,
                child_id,
                mid_short_offset_refinement_steps,
            )?;

            let child_score = sample_tensor_at_single_channel(
                scores,
                height,
                width,
                num_keypoints,
                child_point,
                child_id,
            )?;

            decode_queue.push(KeypointWithScore {
                point: child_point,
                id: child_id,
                score: child_score,
            });
        }
    }
    Ok(())
}

fn pass_keypoint_nms<const N: usize>(
    poses: &[PoseKeypoints<N>],
    keypoint: KeypointWithScore,
    squared_nms_radius: f32,
) -> bool {
    poses.iter().all(|pose| {
        compute_squared_distance(keypoint.point, pose[keypoint.id]) > squared_nms_radius
    })
}

fn find_overlapping_keypoints<const N: usize>(
    pose1: &PoseKeypoints<N>,
    pose2: &PoseKeypoints<N>,
    squared_radius: f32,
    mask: &mut [bool],
) {
    for ((&p1, &p2), m) in pose1.iter().zip(pose2.iter()).zip(mask.iter_mut()) {
        *m = compute_squared_distance(p1, p2) <= squared_radius;
    }
}

fn perform_soft_keypoint_nms<const N: usize>(
    decreasing_indices: &[usize],
    all_keypoint_coords: &[PoseKeypoints<N>],
    all_keypoint_scores: &[PoseKeypointScores<N>],
    num_keypoints: usize,
    squared_nms_radius: f32,
    topk: usize,
    all_instance_scores: &mut Vec<f32>,
) {
    let num_instances = decreasing_indices.len();
    all_instance_scores.resize(num_instances, 0.0);
    // Indicates the occlusion status of the keypoints of the active instance.
    let mut keypoint_occluded = vec![false; num_keypoints];
    // Indices of the keypoints of the active instance in decreasing score value.
    let mut indices = vec![0; num_keypoints];
    for i in 0..num_instances {
        let current_index = decreasing_indices[i];
        // Find the keypoints of the current instance which are overlapping with
        // the corresponding keypoints of the higher-scoring instances and
        // zero-out their contribution to the score of the current instance.
        keypoint_occluded.fill(false);

        for &previous_index in decreasing_indices.iter().take(i) {
            find_overlapping_keypoints(
                &all_keypoint_coords[current_index],
                &all_keypoint_coords[previous_index],
                squared_nms_radius,
                &mut keypoint_occluded,
            );
        }
        // We compute the argsort keypoint indices based on the original keypoint
        // scores, but we do not let them contribute to the instance score if they
        // have been non-maximum suppressed.
        decreasing_arg_sort(&all_keypoint_scores[current_index], &mut indices);

        let total_score = indices
            .iter()
            .take(topk)
            .filter_map(|&index| {
                if !keypoint_occluded[index] {
                    Some(all_keypoint_scores[current_index][index])
                } else {
                    None
                }
            })
            .sum::<f32>();
        all_instance_scores[current_index] = total_score / topk.to_f32().unwrap();
    }
}

impl crate::decode::Decoder for Decoder {
    fn expected_output_tensors(&self) -> usize {
        3
    }

    fn get_decoded_arrays<'a, 'b: 'a>(
        &'a self,
        interp: &'b mut tflite::Interpreter,
        (frame_width, frame_height): (usize, usize),
    ) -> Result<Box<[pose::Pose]>, Error> {
        let recip_output_stride = f32::from(self.output_stride).recip();

        let heatmaps = interp.get_output_tensor(0)?.dequantized()?;
        let shorts = interp
            .get_output_tensor(1)?
            .dequantized_with_scale(recip_output_stride)?;
        let mids = interp
            .get_output_tensor(2)?
            .dequantized_with_scale(recip_output_stride)?;

        let (pose_scores, keypoint_scores, keypoints) = self.decode_all_poses::<NUM_KEYPOINTS>(
            &heatmaps,
            &shorts,
            &mids,
            frame_height,
            frame_width,
        )?;

        crate::decode::reconstruct_from_arrays(
            keypoints.into(),
            keypoint_scores.into(),
            pose_scores.into(),
        )
    }
}

#[cfg(test)]
mod decreasing_arg_sort_tests {
    use super::decreasing_arg_sort;

    #[test]
    fn unsorted_vector() {
        let scores = vec![0.6, 0.897, 0.01, 0.345, 0.28473];
        let mut indices = vec![0; scores.len()];
        decreasing_arg_sort(&scores, &mut indices);
        assert_eq!(indices, vec![1, 0, 3, 4, 2]);
    }

    #[test]
    fn all_same_vector() {
        let scores = vec![0.5; 5];
        let mut indices = vec![0; scores.len()];
        decreasing_arg_sort(&scores, &mut indices);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn negative_vector() {
        let scores = vec![0.6, -0.897, 0.01, 0.345, 0.28473];
        let mut indices = vec![0; scores.len()];
        decreasing_arg_sort(&scores, &mut indices);
        assert_eq!(indices, vec![0, 3, 4, 2, 1]);
    }
}

#[cfg(test)]
mod compute_squared_distance_tests {
    use super::compute_squared_distance;
    use opencv::core::Point2f;

    #[test]
    fn xy_points() {
        let a = Point2f::new(0.5, 0.5);
        let b = Point2f::new(1.0, 1.0);
        assert_eq!(compute_squared_distance(a, b), 0.5);
    }
}

#[cfg(test)]
mod sigmoid_tests {
    use super::sigmoid;

    #[test]
    fn zero() {
        assert_eq!(sigmoid(0.0), 0.5);
    }

    #[test]
    fn twenty() {
        assert_eq!(sigmoid(20.0), 1.0);
    }

    #[test]
    fn negative_five() {
        assert_eq!(sigmoid(-5.0), 0.006692851);
    }
}

#[cfg(test)]
mod log_odds_tests {
    use super::log_odds;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn negative_number() {
        assert!(log_odds(-1.0).is_nan());
    }

    #[test]
    fn zero() {
        assert_eq!(log_odds(0.0), -13.81551);
    }

    #[test]
    fn five_tenths() {
        assert_approx_eq!(log_odds(0.5), 0.000004);
    }
}

#[cfg(test)]
mod build_linear_interpolation_tests {
    use super::build_linear_interpolation;
    use assert_approx_eq::assert_approx_eq;
    use num_traits::cast::ToPrimitive;

    #[test]
    fn build_y_is_valid() {
        let height = 3;
        let y = 0.25 * height.to_f32().unwrap();
        let (y_floor, y_ceil, y_lerp) = build_linear_interpolation(y, height);
        assert_approx_eq!(y_lerp, 0.75);
        assert_eq!(y_floor, 0);
        assert_eq!(y_ceil, 1);
    }
}

#[cfg(test)]
mod build_bilinear_interpolation_tests {
    use super::build_bilinear_interpolation;
    use assert_approx_eq::assert_approx_eq;
    use num_traits::cast::ToPrimitive;

    #[test]
    fn build_xy_is_valid() {
        let height = 3;
        let width = 4;
        let num_channels = 3;
        let y = 0.25 * height.to_f32().unwrap();
        let x = 0.33 * width.to_f32().unwrap();
        let (top_left, top_right, bottom_left, bottom_right, y_lerp, x_lerp) =
            build_bilinear_interpolation(y, x, height, width, num_channels);
        assert_eq!(top_left, 3);
        assert_eq!(top_right, 6);
        assert_eq!(bottom_left, 15);
        assert_eq!(bottom_right, 18);
        assert_approx_eq!(y_lerp, 0.75);
        assert_approx_eq!(x_lerp, 0.32);
    }
}

#[cfg(test)]
mod sample_tensor_at_multiple_channels_tests {
    use super::sample_tensor_at_multiple_channels;
    use assert_approx_eq::assert_approx_eq;
    use num_traits::{cast::ToPrimitive, Zero};
    use ordered_float::NotNan;

    #[test]
    fn sample_is_correct() {
        // Create an input tensor with shape [height, width, num_channels] and
        // values specified as a linear function of the position.
        let height = 3;
        let width = 4;
        let num_channels = 3;
        let size = height * width * num_channels;
        let mut tensor = vec![0.0; size];
        let mut index = 0;
        for y in 0..height {
            for x in 0..width {
                for c in 0..num_channels {
                    tensor[index] = (y + x + c).to_f32().unwrap() + 0.2;
                    index += 1;
                }
            }
        }
        // Sample the tensor at a mid-point position and multiple channels and verify
        // we get the expected results.
        let y = 0.25 * height.to_f32().unwrap();
        let x = 0.33 * width.to_f32().unwrap();

        const CHANNELS: [usize; 4] = [2, 0, 0, 1];
        let n_result_channels = CHANNELS.len();
        let mut result = vec![NotNan::zero(); n_result_channels];

        sample_tensor_at_multiple_channels(
            &tensor,
            height,
            width,
            num_channels,
            y,
            x,
            &CHANNELS,
            &mut result,
        )
        .unwrap();

        assert_eq!(result.len(), CHANNELS.len());

        for (res, ch) in result.into_iter().zip(
            CHANNELS
                .iter()
                .map(|c| NotNan::new(y + x + c.to_f32().unwrap() + 0.2).unwrap()),
        ) {
            assert_approx_eq!(res.into_inner(), ch.into_inner());
        }
    }
}

#[cfg(test)]
mod sample_tensor_at_single_channel_test {
    use super::sample_tensor_at_single_channel;
    use num_traits::cast::ToPrimitive;
    use opencv::core::Point2f;

    #[test]
    fn sample_is_correct() {
        // Create an input tensor with shape [height, width, num_channels] and
        // values specified as a linear function of the position.
        let height = 3;
        let width = 4;
        let num_channels = 3;
        let size = height * width * num_channels;
        let mut tensor = vec![0.0; size];
        let mut index = 0;
        for y in 0..height {
            for x in 0..width {
                for c in 0..num_channels {
                    tensor[index] = (y + x + c).to_f32().unwrap() + 0.1;
                    index += 1;
                }
            }
        }
        // Sample the tensor at a mid-point position and multiple channels and verify
        // we get the expected results.
        let point = Point2f::new(
            0.25 * height.to_f32().unwrap(),
            0.33 * width.to_f32().unwrap(),
        );
        let c = num_channels / 2;

        let result =
            sample_tensor_at_single_channel(&tensor, height, width, num_channels, point, c)
                .unwrap();

        assert_eq!(result, point.y + point.x + c.to_f32().unwrap() + 0.1);
    }
}

#[cfg(test)]
mod find_displaced_position_test {
    use super::find_displaced_position;
    use assert_approx_eq::assert_approx_eq;
    use num_traits::cast::ToPrimitive;
    use opencv::core::Point2f;

    #[test]
    fn position_is_correct_all_zeros() {
        // The short_offsets tensors has size [height, width, num_keypoints * 2].
        // The mid_offsets tensors has size [height, width, 2 * 2 * num_edges].
        let height = 10;
        let width = 8;
        let num_keypoints = 3;
        let num_edges = 2 * (num_keypoints - 1); // Forward-backward chain.
                                                 // Create a short_offsets tensor with all 0s
        let short_offsets = vec![0.0; height * width * num_keypoints * 2];
        // Create a mid_offsets tensor with all 0s
        let mid_offsets = vec![0.0; height * width * 2 * 2 * num_edges];
        let source = Point2f::new(4.1, 3.5);
        let edge_id = 1;
        let target_id = 2;

        let point_results = (0..4)
            .map(|i| {
                find_displaced_position(
                    &short_offsets,
                    &mid_offsets,
                    height,
                    width,
                    num_keypoints,
                    num_edges,
                    source,
                    edge_id,
                    target_id,
                    i,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(point_results, vec![source; 4]);
    }

    #[test]
    fn position_is_correct_all_ones() {
        let height = 10;
        let width = 8;
        let num_keypoints = 3;
        let num_edges = 2 * (num_keypoints - 1); // Forward-backward chain.
                                                 // Create a short_offsets tensor with all 1s
        let short_offsets = vec![1.0; height * width * num_keypoints * 2];
        // Create a mid_offsets tensor with all -1s
        let mid_offsets = vec![-1.0; height * width * 2 * 2 * num_edges];
        let source = Point2f::new(4.1, 3.5);
        let edge_id = 1;
        let target_id = 2;

        for i in 0..4 {
            let point_result = find_displaced_position(
                &short_offsets,
                &mid_offsets,
                height,
                width,
                num_keypoints,
                num_edges,
                source,
                edge_id,
                target_id,
                i,
            )
            .unwrap();

            // We move once by the (-1, -1) mid-offsets array to (y1 - 1, x1 - 1), and
            // then i-times by the (1, 1) short-offsets array, to
            // (y1 + i - 1, x1 + i - 1).
            assert_eq!(point_result.y, source.y + i.to_f32().unwrap() - 1.0);
            assert_eq!(point_result.x, source.x + i.to_f32().unwrap() - 1.0);
        }
    }

    #[test]
    fn position_is_correct() {
        const HEIGHT: usize = 10;
        const WIDTH: usize = 8;
        const NUM_KEYPOINTS: usize = 3;
        const NUM_EDGES: usize = 2 * (NUM_KEYPOINTS - 1);
        let mut short_offsets = vec![0.0; HEIGHT * WIDTH * NUM_KEYPOINTS * 2];
        let short_offsets_max_range = (HEIGHT - 1 + WIDTH - 1 + NUM_KEYPOINTS * 2 - 1)
            .to_f32()
            .unwrap();
        let mut short_offsets_index = 0;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                for c in 0..NUM_KEYPOINTS * 2 {
                    short_offsets[short_offsets_index] =
                        ((y + x + c).to_f32().unwrap() + 0.1) / (short_offsets_max_range + 0.1);
                    short_offsets_index += 1;
                }
            }
        }

        let mid_offsets_max_range = (HEIGHT - 1 + WIDTH - 1 + 2 * 2 * NUM_EDGES - 1)
            .to_f32()
            .unwrap();
        let mut mid_offsets = vec![0.0; HEIGHT * WIDTH * 2 * 2 * NUM_EDGES];
        let mut mid_offsets_index = 0;

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                for c in 0..2 * 2 * NUM_EDGES {
                    mid_offsets[mid_offsets_index] =
                        2.0 * ((y + x + c).to_f32().unwrap() + 0.1) / mid_offsets_max_range - 0.5;
                    mid_offsets_index += 1;
                }
            }
        }

        let source = Point2f::new(3.5, 4.1);
        let edge_id = 1;
        let target_id = 2;

        let expected_points = [
            Point2f::new(3.819355, 4.161290),
            Point2f::new(4.439290, 4.639046),
            Point2f::new(5.111249, 5.168825),
            Point2f::new(5.840163, 5.755558),
        ];

        for (point_result, expected_point) in (0..expected_points.len())
            .map(|i| {
                find_displaced_position(
                    &short_offsets,
                    &mid_offsets,
                    HEIGHT,
                    WIDTH,
                    NUM_KEYPOINTS,
                    NUM_EDGES,
                    source,
                    edge_id,
                    target_id,
                    i,
                )
                .unwrap()
            })
            .zip(expected_points)
        {
            assert_approx_eq!(point_result.x, expected_point.x);
            assert_approx_eq!(point_result.y, expected_point.y);
        }
    }
}

#[cfg(test)]
mod build_adjacency_list_tests {
    use super::{build_adjacency_list, AdjacencyList};

    #[test]
    fn build_is_valid() {
        let mut expected_adjacency_list = AdjacencyList::default();
        expected_adjacency_list.child_ids = vec![
            vec![1, 2, 5, 6],
            vec![3, 0],
            vec![4, 0],
            vec![1],
            vec![2],
            vec![7, 11, 0],
            vec![8, 12, 0],
            vec![9, 5],
            vec![10, 6],
            vec![7],
            vec![8],
            vec![13, 5],
            vec![14, 6],
            vec![15, 11],
            vec![16, 12],
            vec![13],
            vec![14],
        ];
        expected_adjacency_list.edge_ids = vec![
            vec![0, 2, 4, 10],
            vec![1, 16],
            vec![3, 18],
            vec![17],
            vec![19],
            vec![5, 7, 20],
            vec![11, 13, 26],
            vec![6, 21],
            vec![12, 27],
            vec![22],
            vec![28],
            vec![8, 23],
            vec![14, 29],
            vec![9, 24],
            vec![15, 30],
            vec![25],
            vec![31],
        ];
        let adjacency_list = build_adjacency_list();
        assert_eq!(adjacency_list, expected_adjacency_list);
    }
}

#[cfg(test)]
mod backtrack_decode_pose_tests {
    use super::{
        backtrack_decode_pose, AdjacencyList, KeypointWithScore, PoseKeypointScores, PoseKeypoints,
    };
    use assert_approx_eq::assert_approx_eq;
    use opencv::core::Point2f;
    use ordered_float::NotNan;

    #[test]
    fn backtrack_decode_pose_is_correct() {
        const HEIGHT: usize = 10;
        const WIDTH: usize = 8;
        const NUM_KEYPOINTS: usize = 3;
        const NUM_EDGES: usize = 2 * (NUM_KEYPOINTS - 1);

        let scores = [0.8; HEIGHT * WIDTH * NUM_KEYPOINTS];
        let short_offsets = [1.0; HEIGHT * WIDTH * NUM_KEYPOINTS * 2];
        let mut mid_offsets = [-1.0; HEIGHT * WIDTH * 2 * 2 * NUM_EDGES];
        mid_offsets.fill(-1.0);

        let mut adjacency_list = AdjacencyList::default();
        adjacency_list.child_ids = vec![vec![1], vec![2, 0], vec![1]];
        adjacency_list.edge_ids = vec![vec![0], vec![1, 3], vec![2]];
        let mut pose_keypoints = PoseKeypoints::<NUM_KEYPOINTS>::default();
        let mut keypoint_scores = PoseKeypointScores::<NUM_KEYPOINTS>::default();
        let y1 = 7.1;
        let x1 = 5.5;

        let root = KeypointWithScore {
            point: Point2f::new(x1, y1),
            id: 1,
            score: NotNan::new(0.0).unwrap(),
        };

        backtrack_decode_pose(
            &scores,
            &short_offsets,
            &mid_offsets,
            HEIGHT,
            WIDTH,
            NUM_KEYPOINTS,
            NUM_EDGES,
            &root,
            &adjacency_list,
            2,
            &mut pose_keypoints,
            &mut keypoint_scores,
        )
        .unwrap();

        let expected_pose_keypoints = [
            Point2f::new(x1 + 1.0, y1 + 1.0),
            Point2f::new(x1, y1),
            Point2f::new(x1 + 1.0, y1 + 1.0),
        ];

        for ((&coord, expected_coord), score) in pose_keypoints
            .iter()
            .zip(expected_pose_keypoints)
            .zip(keypoint_scores)
        {
            assert_eq!(coord, expected_coord);
            assert_approx_eq!(score, 0.8);
        }
    }
}

#[cfg(test)]
mod build_keypoint_with_score_queue_tests {
    use super::{DecreasingScoreKeypointPriorityQueue, KeypointWithScore};
    use ndarray::{Array, Array3, Array4};
    use num_traits::cast::ToPrimitive;
    use opencv::core::Point2f;
    use ordered_float::NotNan;

    #[test]
    fn build_is_valid_with_threshold() {
        let height = 5;
        let width = 4;
        let num_keypoints = 3;

        let mut scores = Array3::zeros((height, width, num_keypoints));
        let p1 = KeypointWithScore {
            point: Point2f::new(1.0, 2.0),
            id: 1,
            score: NotNan::new(1.0).unwrap(),
        };
        let p2 = KeypointWithScore {
            point: Point2f::new(3.0, 0.0),
            id: 2,
            score: NotNan::new(1.0).unwrap(),
        };
        scores[(
            p1.point.y.to_usize().unwrap(),
            p1.point.x.to_usize().unwrap(),
            p1.id,
        )] = p1.score.into_inner();
        scores[(
            p2.point.y.to_usize().unwrap(),
            p2.point.x.to_usize().unwrap(),
            p2.id,
        )] = p1.score.into_inner();

        let short_offsets = Array4::ones((height, width, num_keypoints, 2));
        let mut queue = DecreasingScoreKeypointPriorityQueue::new();
        queue
            .build_keypoint(
                scores.as_slice().unwrap(),
                short_offsets.as_slice().unwrap(),
                height,
                width,
                num_keypoints,
                0.5,
                1,
            )
            .unwrap();
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn build_is_valid_scores_correct() {
        let height = 5;
        let width = 4;
        let num_keypoints = 3;

        let mut scores = Array::zeros((height, width, num_keypoints));
        let p1 = KeypointWithScore {
            point: Point2f::new(2.0, 1.0),
            id: 2,
            score: NotNan::new(1.0).unwrap(),
        };
        let p2 = KeypointWithScore {
            point: Point2f::new(0.0, 3.0),
            id: 1,
            score: NotNan::new(2.0).unwrap(),
        };
        scores[(
            p1.point.y.to_usize().unwrap(),
            p1.point.x.to_usize().unwrap(),
            p1.id,
        )] = p1.score.into_inner();
        scores[(
            p2.point.y.to_usize().unwrap(),
            p2.point.x.to_usize().unwrap(),
            p2.id,
        )] = p2.score.into_inner();

        let short_offsets = Array4::ones((height, width, num_keypoints, 2));
        let expected_keypoint1 = KeypointWithScore {
            point: p1.point + Point2f::new(1.0, 1.0),
            id: p1.id,
            score: p1.score,
        };
        let expected_keypoint2 = KeypointWithScore {
            point: p2.point + Point2f::new(1.0, 1.0),
            id: p2.id,
            score: p2.score,
        };
        let mut queue = DecreasingScoreKeypointPriorityQueue::new();
        queue
            .build_keypoint(
                scores.as_slice().unwrap(),
                short_offsets.as_slice().unwrap(),
                height,
                width,
                num_keypoints,
                0.5,
                1,
            )
            .unwrap();
        let top = queue.0.pop().unwrap();
        assert_eq!(top.score, expected_keypoint2.score);
        assert_eq!(top.point, expected_keypoint2.point);
        assert_eq!(top.id, expected_keypoint2.id);

        let top = queue.0.pop().unwrap();
        assert_eq!(top.score, expected_keypoint1.score);
        assert_eq!(top.point, expected_keypoint1.point);
        assert_eq!(top.id, expected_keypoint1.id);
    }
}

#[cfg(test)]
mod pass_keypoint_nms_tests {
    use super::{pass_keypoint_nms, KeypointWithScore, PoseKeypoints};
    use opencv::core::Point2f;
    use ordered_float::NotNan;

    #[test]
    fn squared_distance_less_than_squared_nms_radius() {
        let pose_keypoints = [PoseKeypoints::from([
            Point2f::new(0.0, 0.0),
            Point2f::new(1.0, 1.0),
        ])];
        let keypoint1 = KeypointWithScore {
            point: Point2f::new(0.5, 0.5),
            id: 0,
            score: unsafe { NotNan::new_unchecked(0.0) },
        };
        assert!(!pass_keypoint_nms(&pose_keypoints, keypoint1, 0.55));
    }

    #[test]
    fn squared_distance_greater_than_squared_nms_radius() {
        let pose_keypoints = [PoseKeypoints::from([
            Point2f::new(0.0, 0.0),
            Point2f::new(1.0, 1.0),
        ])];
        let keypoint1 = KeypointWithScore {
            point: Point2f::new(0.5, 0.5),
            id: 0,
            score: unsafe { NotNan::new_unchecked(0.0) },
        };
        assert!(pass_keypoint_nms(&pose_keypoints, keypoint1, 0.45));
    }
}

#[cfg(test)]
mod find_overlapping_keypoints_tests {
    use super::{find_overlapping_keypoints, PoseKeypoints};
    use opencv::core::Point2f;

    #[test]
    fn keypoints_result_no_overlap() {
        let pose_keypoints1 = PoseKeypoints::from([Point2f::new(0.0, 0.0), Point2f::new(0.0, 1.0)]);
        let pose_keypoints2 = PoseKeypoints::from([Point2f::new(1.0, 1.0), Point2f::new(1.0, 0.0)]);
        let mut mask = [false, false];
        find_overlapping_keypoints(&pose_keypoints1, &pose_keypoints2, 1.0, &mut mask);
        assert_eq!(mask, [false, false])
    }

    #[test]
    fn keypoints_result_same_points() {
        let pose_keypoints1 = PoseKeypoints::from([Point2f::new(0.0, 0.0), Point2f::new(0.0, 1.0)]);
        let pose_keypoints2 = PoseKeypoints::from([Point2f::new(0.0, 0.0), Point2f::new(0.0, 1.0)]);
        let mut mask = [false, false];
        find_overlapping_keypoints(&pose_keypoints1, &pose_keypoints2, 1.0, &mut mask);
        assert_eq!(mask, [true, true])
    }

    #[test]
    fn keypoints_result_overlap() {
        let pose_keypoints1 = PoseKeypoints::from([Point2f::new(0.0, 0.0), Point2f::new(1.0, 1.0)]);
        let pose_keypoints2 = PoseKeypoints::from([Point2f::new(0.0, 0.9), Point2f::new(2.0, 2.0)]);
        let mut mask = [false, false];
        find_overlapping_keypoints(&pose_keypoints1, &pose_keypoints2, 1.0, &mut mask);
        assert_eq!(mask, [true, false])
    }
}

#[cfg(test)]
mod perform_soft_keypoint_nms_tests {
    use super::{perform_soft_keypoint_nms, PoseKeypointScores, PoseKeypoints};
    use assert_approx_eq::assert_approx_eq;
    use opencv::core::Point2f;

    fn test_soft_keypoint_nms(squared_nms_radius: f32, topk: usize) -> Vec<f32> {
        let num_keypoints = 2;
        let mut all_instance_scores = vec![0_f32; num_keypoints];
        let all_keypoint_coords = [
            PoseKeypoints::from([Point2f::new(0.0, 0.0), Point2f::new(1.0, 1.0)]),
            PoseKeypoints::from([Point2f::new(1.0, 0.0), Point2f::new(2.0, 2.0)]),
        ];
        let all_keypoint_scores = [
            PoseKeypointScores::from([0.1, 0.2]),
            PoseKeypointScores::from([0.3, 0.4]),
        ];
        let decreasing_indices = vec![1, 0];
        perform_soft_keypoint_nms(
            &decreasing_indices,
            &all_keypoint_coords,
            &all_keypoint_scores,
            num_keypoints,
            squared_nms_radius,
            topk,
            &mut all_instance_scores,
        );
        all_instance_scores
    }

    macro_rules! assert_approx_eq_iters {
        ($left:expr, $right:expr) => {
            for (lhs, rhs) in $left.into_iter().zip($right) {
                assert_approx_eq!(lhs, rhs);
            }
        };
    }

    #[test]
    fn perform_soft_keypoint_nms_average() {
        assert_approx_eq_iters!(test_soft_keypoint_nms(0.9, 2), vec![0.15, 0.35]);
        assert_approx_eq_iters!(test_soft_keypoint_nms(1.5, 2), vec![0.1, 0.35]);
        assert_approx_eq_iters!(test_soft_keypoint_nms(2.1, 2), vec![0.0, 0.35]);
    }

    #[test]
    fn perform_soft_keypoint_nms_maximum() {
        assert_approx_eq_iters!(test_soft_keypoint_nms(0.9, 1), vec![0.2, 0.4]);
        assert_approx_eq_iters!(test_soft_keypoint_nms(1.5, 1), vec![0.2, 0.4]);
        assert_approx_eq_iters!(test_soft_keypoint_nms(2.1, 1), vec![0.0, 0.4]);
    }
}
