use crate::{error::Error, pose, tflite};
use adjacency_list::{build_adjacency_list, AdjacencyList};
use bitvec::{bitvec, prelude::BitSlice};
use keypoint_priority_queue::KeypointPriorityQueue;
use keypoint_with_score::KeypointWithScore;
use ndarray::{Array, Array1, Array2, Array3};
use num_traits::{cast::ToPrimitive, Zero};
use ordered_float::NotNan;
use point::Point;
use std::ops::DerefMut;

mod adjacency_list;
mod keypoint_priority_queue;
mod keypoint_with_score;
mod point;

const DEFAULT_OUTPUT_STRIDE: &str = "16";
const DEFAULT_MAX_POSE_DETECTIONS: &str = "100";
const DEFAULT_SCORE_THRESHOLD: &str = "0.2";
const DEFAULT_NMS_RADIUS: &str = "10";
const DEFAULT_MID_SHORT_OFFSET_REFINEMENT_STEPS: &str = "5";

#[derive(Debug, Clone, Copy, structopt::StructOpt)]
pub(crate) struct Decoder {
    #[structopt(short, long, default_value = DEFAULT_OUTPUT_STRIDE)]
    pub(crate) output_stride: u8,
    #[structopt(short = "-d", long, default_value = DEFAULT_MAX_POSE_DETECTIONS)]
    pub(crate) max_pose_detections: usize,
    #[structopt(short, long, default_value = DEFAULT_SCORE_THRESHOLD)]
    pub(crate) score_threshold: f32,
    #[structopt(short, long, default_value = DEFAULT_NMS_RADIUS)]
    pub(crate) nms_radius: usize,
    #[structopt(short = "-r", long, default_value = DEFAULT_MID_SHORT_OFFSET_REFINEMENT_STEPS)]
    pub(crate) mid_short_offset_refinement_steps: usize,
}

type PoseKeypoints<const N: usize> = [Point; N];
type PoseKeypointScores<const N: usize> = [f32; N];

impl Default for Decoder {
    fn default() -> Self {
        Self {
            output_stride: DEFAULT_OUTPUT_STRIDE.parse().unwrap(),
            max_pose_detections: DEFAULT_MAX_POSE_DETECTIONS.parse().unwrap(),
            score_threshold: DEFAULT_SCORE_THRESHOLD.parse().unwrap(),
            nms_radius: DEFAULT_NMS_RADIUS.parse().unwrap(),
            mid_short_offset_refinement_steps: DEFAULT_MID_SHORT_OFFSET_REFINEMENT_STEPS
                .parse()
                .unwrap(),
        }
    }
}

const NUM_EDGES: usize = pose::constants::EDGE_LIST.len() / 2;

fn decreasing_arg_sort(scores: &[f32], indices: &mut [usize]) {
    indices.iter_mut().enumerate().for_each(|(i, dst)| {
        *dst = i;
    });
    indices.sort_by_key(|&i| std::cmp::Reverse(NotNan::new(scores[i]).expect("got NaN score")))
}

impl Decoder {
    #[allow(clippy::type_complexity)]
    fn decode_all_poses<const NUM_KEYPOINTS: usize>(
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
        let mut pose_keypoint_scores = vec![[0.0; NUM_KEYPOINTS]; max_pose_detections];
        let mut pose_keypoints = vec![[Default::default(); NUM_KEYPOINTS]; max_pose_detections];

        let local_maximum_radius = pose::constants::LOCAL_MAXIMUM_RADIUS;
        let min_score_logit = log_odds(score_threshold);

        let mut queue = KeypointPriorityQueue::new();

        queue.build_keypoint::<NUM_KEYPOINTS>(
            scores,
            short_offsets,
            height,
            width,
            min_score_logit,
            local_maximum_radius,
        )?;

        let adjacency_list = build_adjacency_list(NUM_KEYPOINTS)?;
        let topk = NUM_KEYPOINTS;
        let mut indices = [0; NUM_KEYPOINTS];
        let mut pose_counter = 0;

        let mut all_instance_scores = Vec::with_capacity(max_pose_detections);
        let mut scratch_poses = vec![[Default::default(); NUM_KEYPOINTS]; max_pose_detections];
        let mut scratch_keypoint_scores = vec![[0.0; NUM_KEYPOINTS]; max_pose_detections];

        while let Some(root) = queue.pop() {
            if pose_counter >= max_pose_detections {
                break;
            }

            if !pass_keypoint_nms(&pose_keypoints[..pose_counter], root, nms_radius.powi(2)) {
                continue;
            }

            let next_pose = &mut scratch_poses[pose_counter];
            next_pose.fill(Point::new(-1.0, -1.0)?);

            let next_scores = &mut scratch_keypoint_scores[pose_counter];
            next_scores.fill(-1e5);

            backtrack_decode_pose(
                scores,
                short_offsets,
                mid_offsets,
                height,
                width,
                NUM_KEYPOINTS,
                NUM_EDGES,
                &root,
                &adjacency_list,
                self.mid_short_offset_refinement_steps,
                next_pose,
                next_scores,
            )?;

            next_scores.iter_mut().try_for_each(|v| {
                *v = sigmoid(*v)?;
                Ok(())
            })?;

            decreasing_arg_sort(next_scores, &mut indices);

            let instance_score = indices
                .iter()
                .take(topk)
                .map(|&i| next_scores[i])
                .sum::<f32>()
                / topk.to_f32().ok_or(Error::ConvertToF32)?;
            if instance_score >= score_threshold {
                pose_counter += 1;
                all_instance_scores.push(instance_score);
            }
        }

        let mut decreasing_indices = vec![0; all_instance_scores.len()];
        decreasing_arg_sort(&all_instance_scores, &mut decreasing_indices);

        let all_instance_scores = perform_soft_keypoint_nms(
            &decreasing_indices,
            &scratch_poses,
            &scratch_keypoint_scores,
            nms_radius.powi(2),
            topk,
            all_instance_scores,
        )?;

        decreasing_arg_sort(&all_instance_scores, &mut decreasing_indices);

        pose_counter = 0;

        for index in decreasing_indices
            .into_iter()
            .take_while(|&index| all_instance_scores[index] >= score_threshold)
        {
            for k in 0..NUM_KEYPOINTS {
                *pose_keypoints[pose_counter][k].y_mut() =
                    scratch_poses[index][k].y() * output_stride;
                *pose_keypoints[pose_counter][k].x_mut() =
                    scratch_poses[index][k].x() * output_stride;
            }

            pose_keypoint_scores[pose_counter] = scratch_keypoint_scores[index];
            pose_scores[pose_counter] = all_instance_scores[index];
            pose_counter += 1;
        }

        let mut pose_keypoint_scores_arr = Array::zeros((max_pose_detections, NUM_KEYPOINTS));
        let mut pose_keypoints_arr = Array::zeros((max_pose_detections, NUM_KEYPOINTS, 2));

        for (i, (keypoints, keypoint_scores)) in pose_keypoints
            .into_iter()
            .zip(pose_keypoint_scores)
            .enumerate()
        {
            for (j, (point, score)) in keypoints.iter().zip(keypoint_scores).enumerate() {
                pose_keypoint_scores_arr[(i, j)] = score;
                pose_keypoints_arr[(i, j, 0)] = point.y();
                pose_keypoints_arr[(i, j, 1)] = point.x();
            }
        }
        Ok((
            pose_scores.into(),
            pose_keypoint_scores_arr,
            pose_keypoints_arr,
        ))
    }
}

#[inline]
fn sigmoid(x: f32) -> Result<f32, Error> {
    let value = 1.0 / (1.0 + (-x).exp());
    Ok(NotNan::new(value)
        .map_err(|e| Error::ConstructNotNan(e, value))?
        .into_inner())
}

#[inline]
fn log_odds(x: f32) -> f32 {
    -(1.0 / (x + 1e-6) - 1.0).ln()
}

fn build_linear_interpolation(x: f32, n: usize) -> Result<(usize, usize, f32), Error> {
    let x_proj = x.clamp(0.0, n.to_f32().ok_or(Error::ConvertToF32)? - 1.0);
    let floor_f = x_proj.floor();
    let ceil_f = x_proj.ceil();
    Ok((
        floor_f.to_usize().ok_or(Error::ConvertToUSize)?,
        ceil_f.to_usize().ok_or(Error::ConvertToUSize)?,
        x - floor_f,
    ))
}

fn build_bilinear_interpolation(
    y: f32,
    x: f32,
    height: usize,
    width: usize,
    num_channels: usize,
) -> Result<(usize, usize, usize, usize, f32, f32), Error> {
    let (y_floor, y_ceil, y_lerp) = build_linear_interpolation(y, height)?;
    let (x_floor, x_ceil, x_lerp) = build_linear_interpolation(x, width)?;
    let top_left = (y_floor * width + x_floor) * num_channels;
    let top_right = (y_floor * width + x_ceil) * num_channels;
    let bottom_left = (y_ceil * width + x_floor) * num_channels;
    let bottom_right = (y_ceil * width + x_ceil) * num_channels;
    Ok((
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        y_lerp,
        x_lerp,
    ))
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
        build_bilinear_interpolation(y, x, height, width, num_channels)?;
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
    point: Point,
    c: usize,
) -> Result<NotNan<f32>, Error> {
    let mut result = [NotNan::zero(); 1];
    let c = [c];
    sample_tensor_at_multiple_channels(
        tensor,
        height,
        width,
        num_channels,
        point.y(),
        point.x(),
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
    source: Point,
    edge_id: usize,
    target_id: usize,
    mid_short_offset_refinement_steps: usize,
) -> Result<Point, Error> {
    let mut y = source.y();
    let mut x = source.x();
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
    let float_height = height.to_f32().ok_or(Error::ConvertToF32)?;
    let float_width = width.to_f32().ok_or(Error::ConvertToF32)?;
    y = (y + offsets[0].into_inner()).clamp(0.0, float_height - 1.0);
    x = (x + offsets[1].into_inner()).clamp(0.0, float_width - 1.0);
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
        y = (y + offsets[0].into_inner()).clamp(0.0, float_height - 1.0);
        x = (x + offsets[1].into_inner()).clamp(0.0, float_width - 1.0);
    }

    Point::new(x, y)
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
    let mut decode_queue = KeypointPriorityQueue::new();
    decode_queue.push(KeypointWithScore {
        point: root.point,
        id: root.id,
        score: root_score,
    });

    // Keeps track of the keypoints whose position has already been decoded.
    let mut keypoint_decoded = bitvec![0; num_keypoints];

    // The top element in the queue is the next keypoint to be processed.
    while let Some(KeypointWithScore { point, id, score }) = decode_queue.pop() {
        if keypoint_decoded[id] {
            continue;
        }

        pose_keypoints[id] = point;
        keypoint_scores[id] = score.into_inner();
        keypoint_decoded.set(id, true);

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
    KeypointWithScore { point, id, .. }: KeypointWithScore,
    squared_nms_radius: f32,
) -> bool {
    poses
        .iter()
        .all(|pose| point.squared_distance(pose[id]) > squared_nms_radius)
}

fn find_overlapping_keypoints<const N: usize>(
    pose1: &PoseKeypoints<N>,
    pose2: &PoseKeypoints<N>,
    squared_radius: f32,
    mask: &mut BitSlice,
) {
    pose1
        .iter()
        .zip(pose2)
        .enumerate()
        .for_each(|(i, (&p1, &p2))| mask.set(i, p1.squared_distance(p2) <= squared_radius))
}

fn perform_soft_keypoint_nms<const N: usize>(
    decreasing_indices: &[usize],
    all_keypoint_coords: &[PoseKeypoints<N>],
    all_keypoint_scores: &[PoseKeypointScores<N>],
    squared_nms_radius: f32,
    topk: usize,
    mut all_instance_scores: Vec<f32>,
) -> Result<Vec<f32>, Error> {
    let num_instances = decreasing_indices.len();
    all_instance_scores.resize(num_instances, 0.0);
    // Indicates the occlusion status of the keypoints of the active instance.
    let mut keypoint_occluded = bitvec![0; N];
    // Indices of the keypoints of the active instance in decreasing score value.
    let mut indices = [0; N];
    let topk_float = topk.to_f32().ok_or(Error::ConvertToF32)?;
    for (i, &current_index) in decreasing_indices.iter().take(num_instances).enumerate() {
        // Find the keypoints of the current instance which are overlapping with
        // the corresponding keypoints of the higher-scoring instances and
        // zero-out their contribution to the score of the current instance.
        keypoint_occluded.set_all(false);

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

        all_instance_scores[current_index] = total_score / topk_float;
    }
    Ok(all_instance_scores)
}

impl crate::decode::Decoder for Decoder {
    fn expected_output_tensors(&self) -> usize {
        3
    }

    fn decode_output<I>(&self, interp: I) -> Result<Box<[pose::Pose]>, Error>
    where
        I: DerefMut<Target = tflite::Interpreter>,
    {
        use pose::NUM_KEYPOINTS;

        let recip_output_stride = f32::from(self.output_stride).recip();

        let mut heatmap_tensor = interp.get_output_tensor_by_name("float_heatmaps")?;
        let frame_height = heatmap_tensor.dim(1)?;
        let frame_width = heatmap_tensor.dim(2)?;
        let heatmaps = heatmap_tensor.dequantized()?;
        let shorts = interp
            .get_output_tensor_by_name("float_short_offsets")?
            .dequantized_with_scale(recip_output_stride)?;
        let mids = interp
            .get_output_tensor_by_name("float_mid_offsets")?
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
mod tests {
    use super::*;

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

    mod sigmoid_tests {
        use super::sigmoid;
        use assert_approx_eq::assert_approx_eq;

        #[test]
        fn zero() {
            assert_approx_eq!(sigmoid(0.0).unwrap(), 0.5);
        }

        #[test]
        fn twenty() {
            assert_approx_eq!(sigmoid(20.0).unwrap(), 1.0);
        }

        #[test]
        fn negative_five() {
            assert_approx_eq!(sigmoid(-5.0).unwrap(), 0.006692851);
        }
    }

    mod log_odds_tests {
        use super::log_odds;
        use assert_approx_eq::assert_approx_eq;

        #[test]
        fn negative_number() {
            assert!(log_odds(-1.0).is_nan());
        }

        #[test]
        fn zero() {
            assert_approx_eq!(log_odds(0.0), -13.81551);
        }

        #[test]
        fn five_tenths() {
            assert_approx_eq!(log_odds(0.5), 0.000004);
        }
    }

    mod build_linear_interpolation_tests {
        use super::build_linear_interpolation;
        use assert_approx_eq::assert_approx_eq;
        use num_traits::cast::ToPrimitive;

        #[test]
        fn build_y_is_valid() {
            const HEIGHT: usize = 3;
            let y = 0.25 * HEIGHT.to_f32().unwrap();
            let (y_floor, y_ceil, y_lerp) = build_linear_interpolation(y, HEIGHT).unwrap();
            assert_approx_eq!(y_lerp, 0.75);
            assert_eq!(y_floor, 0);
            assert_eq!(y_ceil, 1);
        }
    }

    mod build_bilinear_interpolation_tests {
        use super::build_bilinear_interpolation;
        use assert_approx_eq::assert_approx_eq;
        use num_traits::cast::ToPrimitive;

        #[test]
        fn build_xy_is_valid() {
            const HEIGHT: usize = 3;
            const WIDTH: usize = 4;
            const NUM_CHANNELS: usize = 3;
            let y = 0.25 * HEIGHT.to_f32().unwrap();
            let x = 0.33 * WIDTH.to_f32().unwrap();
            let (top_left, top_right, bottom_left, bottom_right, y_lerp, x_lerp) =
                build_bilinear_interpolation(y, x, HEIGHT, WIDTH, NUM_CHANNELS).unwrap();
            assert_eq!(top_left, 3);
            assert_eq!(top_right, 6);
            assert_eq!(bottom_left, 15);
            assert_eq!(bottom_right, 18);
            assert_approx_eq!(y_lerp, 0.75);
            assert_approx_eq!(x_lerp, 0.32);
        }
    }

    mod sample_tensor_at_multiple_channels_tests {
        use super::sample_tensor_at_multiple_channels;
        use assert_approx_eq::assert_approx_eq;
        use num_traits::{cast::ToPrimitive, Zero};
        use ordered_float::NotNan;

        #[test]
        fn sample_is_correct() {
            // Create an input tensor with shape [height, width, num_channels] and
            // values specified as a linear function of the position.
            const HEIGHT: usize = 3;
            const WIDTH: usize = 4;
            const NUM_CHANNELS: usize = 3;
            let mut tensor = [0.0; HEIGHT * WIDTH * NUM_CHANNELS];
            let mut index = 0;
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    for c in 0..NUM_CHANNELS {
                        tensor[index] = (y + x + c).to_f32().unwrap() + 0.2;
                        index += 1;
                    }
                }
            }
            // Sample the tensor at a mid-point position and multiple channels and verify
            // we get the expected results.
            let y = 0.25 * HEIGHT.to_f32().unwrap();
            let x = 0.33 * WIDTH.to_f32().unwrap();

            const CHANNELS: [usize; 4] = [2, 0, 0, 1];
            const N_RESULT_CHANNELS: usize = CHANNELS.len();
            let mut result = [NotNan::zero(); N_RESULT_CHANNELS];

            sample_tensor_at_multiple_channels(
                &tensor,
                HEIGHT,
                WIDTH,
                NUM_CHANNELS,
                y,
                x,
                &CHANNELS,
                &mut result,
            )
            .unwrap();

            assert_eq!(result.len(), N_RESULT_CHANNELS);

            for (res, ch) in result.iter().zip(
                CHANNELS
                    .iter()
                    .map(|c| NotNan::new(y + x + c.to_f32().unwrap() + 0.2).unwrap()),
            ) {
                assert_approx_eq!(res.into_inner(), ch.into_inner());
            }
        }
    }

    mod sample_tensor_at_single_channel_test {
        use super::{sample_tensor_at_single_channel, Point};
        use assert_approx_eq::assert_approx_eq;
        use num_traits::cast::ToPrimitive;

        #[test]
        fn sample_is_correct() {
            // Create an input tensor with shape [height, width, num_channels] and
            // values specified as a linear function of the position.
            const HEIGHT: usize = 3;
            const WIDTH: usize = 4;
            const NUM_CHANNELS: usize = 3;
            let mut tensor = [0.0; HEIGHT * WIDTH * NUM_CHANNELS];
            let mut index = 0;
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    for c in 0..NUM_CHANNELS {
                        tensor[index] = (y + x + c).to_f32().unwrap() + 0.1;
                        index += 1;
                    }
                }
            }
            // Sample the tensor at a mid-point position and multiple channels and verify
            // we get the expected results.
            let point = Point::new(
                0.25 * HEIGHT.to_f32().unwrap(),
                0.33 * WIDTH.to_f32().unwrap(),
            )
            .unwrap();

            const C: usize = NUM_CHANNELS / 2;

            let result =
                sample_tensor_at_single_channel(&tensor, HEIGHT, WIDTH, NUM_CHANNELS, point, C)
                    .unwrap();

            assert_approx_eq!(result, point.y() + point.x() + C.to_f32().unwrap() + 0.1);
        }
    }

    mod find_displaced_position_test {
        use super::{find_displaced_position, Point};
        use assert_approx_eq::assert_approx_eq;
        use num_traits::cast::ToPrimitive;

        #[test]
        fn position_is_correct_all_zeros() {
            // The short_offsets tensors has size [height, width, num_keypoints * 2].
            // The mid_offsets tensors has size [height, width, 2 * 2 * num_edges].
            const HEIGHT: usize = 10;
            const WIDTH: usize = 8;
            const NUM_KEYPOINTS: usize = 3;
            const NUM_EDGES: usize = 2 * (NUM_KEYPOINTS - 1); // Forward-backward chain.
                                                              // Create a short_offsets tensor with all 0s
            let short_offsets = [0.0; HEIGHT * WIDTH * NUM_KEYPOINTS * 2];
            // Create a mid_offsets tensor with all 0s
            let mid_offsets = [0.0; HEIGHT * WIDTH * 2 * 2 * NUM_EDGES];
            let source = Point::new(4.1, 3.5).unwrap();
            let edge_id = 1;
            let target_id = 2;

            let point_results = (0..4)
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
                })
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(point_results, [source; 4]);
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
            let source = Point::new(4.1, 3.5).unwrap();
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
                assert_approx_eq!(point_result.y(), source.y() + i.to_f32().unwrap() - 1.0);
                assert_approx_eq!(point_result.x(), source.x() + i.to_f32().unwrap() - 1.0);
            }
        }

        #[test]
        fn position_is_correct() {
            const HEIGHT: usize = 10;
            const WIDTH: usize = 8;
            const NUM_KEYPOINTS: usize = 3;
            const NUM_EDGES: usize = 2 * (NUM_KEYPOINTS - 1);
            let mut short_offsets = [0.0; HEIGHT * WIDTH * NUM_KEYPOINTS * 2];
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
            let mut mid_offsets = [0.0; HEIGHT * WIDTH * 2 * 2 * NUM_EDGES];
            let mut mid_offsets_index = 0;

            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    for c in 0..2 * 2 * NUM_EDGES {
                        mid_offsets[mid_offsets_index] =
                            2.0 * ((y + x + c).to_f32().unwrap() + 0.1) / mid_offsets_max_range
                                - 0.5;
                        mid_offsets_index += 1;
                    }
                }
            }

            let source = Point::new(3.5, 4.1).unwrap();
            const EDGE_ID: usize = 1;
            const TARGET_ID: usize = 2;

            let expected_points = [
                #[allow(clippy::excessive_precision)]
                Point::new(3.819355, 4.161290).unwrap(),
                #[allow(clippy::excessive_precision)]
                Point::new(4.439290, 4.639046).unwrap(),
                Point::new(5.111249, 5.168825).unwrap(),
                Point::new(5.840163, 5.755558).unwrap(),
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
                        EDGE_ID,
                        TARGET_ID,
                        i,
                    )
                    .unwrap()
                })
                .zip(expected_points)
            {
                assert_approx_eq!(point_result.x(), expected_point.x());
                assert_approx_eq!(point_result.y(), expected_point.y());
            }
        }
    }

    mod backtrack_decode_pose_tests {
        use super::{
            backtrack_decode_pose, AdjacencyList, KeypointWithScore, Point, PoseKeypointScores,
            PoseKeypoints,
        };
        use assert_approx_eq::assert_approx_eq;
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

            let adjacency_list = AdjacencyList {
                child_ids: vec![vec![1], vec![2, 0], vec![1]],
                edge_ids: vec![vec![0], vec![1, 3], vec![2]],
            };
            let mut pose_keypoints = PoseKeypoints::<NUM_KEYPOINTS>::default();
            let mut keypoint_scores = PoseKeypointScores::<NUM_KEYPOINTS>::default();
            const Y1: f32 = 7.1;
            const X1: f32 = 5.5;

            let root = KeypointWithScore {
                point: Point::new(X1, Y1).unwrap(),
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
                Point::new(X1 + 1.0, Y1 + 1.0).unwrap(),
                Point::new(X1, Y1).unwrap(),
                Point::new(X1 + 1.0, Y1 + 1.0).unwrap(),
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

    mod pass_keypoint_nms_tests {
        use super::{pass_keypoint_nms, KeypointWithScore, Point};
        use ordered_float::NotNan;

        #[test]
        fn squared_distance_less_than_squared_nms_radius() {
            let pose_keypoints = [[Point::new(0.0, 0.0).unwrap(), Point::new(1.0, 1.0).unwrap()]];
            let keypoint1 = KeypointWithScore {
                point: Point::new(0.5, 0.5).unwrap(),
                id: 0,
                score: unsafe { NotNan::new_unchecked(0.0) },
            };
            assert!(!pass_keypoint_nms(&pose_keypoints, keypoint1, 0.55));
        }

        #[test]
        fn squared_distance_greater_than_squared_nms_radius() {
            let pose_keypoints = [[Point::new(0.0, 0.0).unwrap(), Point::new(1.0, 1.0).unwrap()]];
            let keypoint1 = KeypointWithScore {
                point: Point::new(0.5, 0.5).unwrap(),
                id: 0,
                score: unsafe { NotNan::new_unchecked(0.0) },
            };
            assert!(pass_keypoint_nms(&pose_keypoints, keypoint1, 0.45));
        }
    }

    mod find_overlapping_keypoints_tests {
        use super::{find_overlapping_keypoints, Point};
        use bitvec::prelude::*;

        #[test]
        fn keypoints_result_no_overlap() {
            let pose_keypoints1 = [Point::new(0.0, 0.0).unwrap(), Point::new(0.0, 1.0).unwrap()];
            let pose_keypoints2 = [Point::new(1.0, 1.0).unwrap(), Point::new(1.0, 0.0).unwrap()];
            let mask = bits![mut 0, 0];
            find_overlapping_keypoints(&pose_keypoints1, &pose_keypoints2, 1.0, mask);
            assert_eq!(mask, bits![0, 0])
        }

        #[test]
        fn keypoints_result_same_points() {
            let pose_keypoints1 = [Point::new(0.0, 0.0).unwrap(), Point::new(0.0, 1.0).unwrap()];
            let pose_keypoints2 = [Point::new(0.0, 0.0).unwrap(), Point::new(0.0, 1.0).unwrap()];
            let mask = bits![mut 0, 0];
            find_overlapping_keypoints(&pose_keypoints1, &pose_keypoints2, 1.0, mask);
            assert_eq!(mask, bits![1, 1])
        }

        #[test]
        fn keypoints_result_overlap() {
            let pose_keypoints1 = [Point::new(0.0, 0.0).unwrap(), Point::new(1.0, 1.0).unwrap()];
            let pose_keypoints2 = [Point::new(0.0, 0.9).unwrap(), Point::new(2.0, 2.0).unwrap()];
            let mask = bits![mut 0; 2];
            find_overlapping_keypoints(&pose_keypoints1, &pose_keypoints2, 1.0, mask);
            assert_eq!(mask, bits![1, 0])
        }
    }

    mod perform_soft_keypoint_nms_tests {
        use super::{perform_soft_keypoint_nms, Point};
        use assert_approx_eq::assert_approx_eq;

        fn test_soft_keypoint_nms(squared_nms_radius: f32, topk: usize) -> Vec<f32> {
            const NUM_KEYPOINTS: usize = 2;
            let all_instance_scores = vec![0.0; NUM_KEYPOINTS];
            let all_keypoint_coords = [
                [Point::new(0.0, 0.0).unwrap(), Point::new(1.0, 1.0).unwrap()],
                [Point::new(1.0, 0.0).unwrap(), Point::new(2.0, 2.0).unwrap()],
            ];
            let all_keypoint_scores = [[0.1, 0.2], [0.3, 0.4]];
            let decreasing_indices = [1, 0];
            perform_soft_keypoint_nms::<NUM_KEYPOINTS>(
                &decreasing_indices,
                &all_keypoint_coords,
                &all_keypoint_scores,
                squared_nms_radius,
                topk,
                all_instance_scores,
            )
            .unwrap()
        }

        fn assert_approx_eq_iters<I, J>(left: I, right: J)
        where
            I: IntoIterator<Item = f32>,
            J: IntoIterator<Item = f32>,
        {
            for (lhs, rhs) in left.into_iter().zip(right) {
                assert_approx_eq!(lhs, rhs);
            }
        }

        #[test]
        fn perform_soft_keypoint_nms_average() {
            assert_approx_eq_iters(test_soft_keypoint_nms(0.9, 2), [0.15, 0.35]);
            assert_approx_eq_iters(test_soft_keypoint_nms(1.5, 2), [0.1, 0.35]);
            assert_approx_eq_iters(test_soft_keypoint_nms(2.1, 2), [0.0, 0.35]);
        }

        #[test]
        fn perform_soft_keypoint_nms_maximum() {
            assert_approx_eq_iters(test_soft_keypoint_nms(0.9, 1), [0.2, 0.4]);
            assert_approx_eq_iters(test_soft_keypoint_nms(1.5, 1), [0.2, 0.4]);
            assert_approx_eq_iters(test_soft_keypoint_nms(2.1, 1), [0.0, 0.4]);
        }
    }
}
