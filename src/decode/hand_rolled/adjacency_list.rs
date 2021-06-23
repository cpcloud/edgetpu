use crate::{error::Error, pose};

#[derive(Debug, PartialEq, PartialOrd, Eq, Ord)]
pub(super) struct AdjacencyList {
    pub(super) child_ids: Vec<Vec<usize>>,
    pub(super) edge_ids: Vec<Vec<usize>>,
}

impl AdjacencyList {
    pub(super) fn new(n: usize) -> Self {
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

pub(super) fn build_adjacency_list(n: usize) -> Result<AdjacencyList, Error> {
    let mut adjacency_list = AdjacencyList::new(n);
    for (k, (parent, child)) in pose::constants::EDGE_LIST.iter().enumerate() {
        let parent_id = parent.idx()?;
        let child_id = child.idx()?;
        adjacency_list.child_ids[parent_id].push(child_id);
        adjacency_list.edge_ids[parent_id].push(k);
    }
    Ok(adjacency_list)
}

#[cfg(test)]
mod tests {
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
        let adjacency_list = build_adjacency_list(crate::pose::NUM_KEYPOINTS).unwrap();
        assert_eq!(adjacency_list, expected_adjacency_list);
    }
}
