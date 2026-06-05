use super::partition::NUM_PARTITIONS;
use super::*;

#[test]
fn test_stable_scatter_by_partition_returns_complete_stable_partition_permutation() {
    let n_rows = 180_000;
    let hashes: Vec<u64> = (0..n_rows)
        .map(|row| (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();

    let (perm, offsets) = stable_scatter_by_partition(&hashes);
    assert_eq!(perm.len(), n_rows);
    assert_eq!(offsets.len(), NUM_PARTITIONS + 1);
    assert_eq!(offsets[0], 0);
    assert_eq!(offsets[NUM_PARTITIONS], n_rows);

    let mut seen = vec![false; n_rows];
    for p in 0..NUM_PARTITIONS {
        let start = offsets[p];
        let end = offsets[p + 1];

        for &row in &perm[start..end] {
            assert!(
                row < n_rows,
                "row index {row} should stay within 0..{n_rows}"
            );
            assert!(
                !seen[row],
                "row index {row} should appear only once in the permutation"
            );
            seen[row] = true;
            assert_eq!(
                hashes[row] as usize & (NUM_PARTITIONS - 1),
                p,
                "row {row} should land in partition {p}"
            );
        }

        let slice = &perm[start..end];
        assert!(
            slice.windows(2).all(|window| window[0] < window[1]),
            "partition {p} should preserve source row order"
        );
    }

    assert!(
        seen.into_iter().all(|present| present),
        "perm should contain every source row exactly once"
    );
}
