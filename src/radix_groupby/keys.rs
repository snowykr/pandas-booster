use smallvec::SmallVec;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

const INLINE_KEYS: usize = 10;

#[derive(Clone, Debug)]
pub struct CompositeKey(SmallVec<[i64; INLINE_KEYS]>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FixedKey<const N: usize>(pub [i64; N]);

impl CompositeKey {
    #[inline]
    fn with_capacity(cap: usize) -> Self {
        Self(SmallVec::with_capacity(cap))
    }

    #[inline]
    fn push(&mut self, val: i64) {
        self.0.push(val);
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = &i64> {
        self.0.iter()
    }
}

impl PartialEq for CompositeKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for CompositeKey {}

impl Hash for CompositeKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &v in &self.0 {
            v.hash(state);
        }
    }
}

impl PartialOrd for CompositeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompositeKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.iter().cmp(other.0.iter())
    }
}

#[inline]
fn compute_hash(key_slices: &[&[i64]], row: usize) -> u64 {
    use ahash::AHasher;
    let mut hasher = AHasher::default();
    for col in key_slices {
        col[row].hash(&mut hasher);
    }
    hasher.finish()
}

fn extract_key(key_slices: &[&[i64]], row: usize) -> CompositeKey {
    let mut key = CompositeKey::with_capacity(key_slices.len());
    for col in key_slices {
        key.push(col[row]);
    }
    key
}

#[inline]
fn compute_hash_fixed<const N: usize>(key_slices: &[&[i64]], row: usize) -> u64 {
    use ahash::AHasher;
    let mut hasher = AHasher::default();
    for col in &key_slices[..N] {
        col[row].hash(&mut hasher);
    }
    hasher.finish()
}

#[inline]
fn extract_key_fixed<const N: usize>(key_slices: &[&[i64]], row: usize) -> FixedKey<N> {
    let mut arr = [0i64; N];
    for (dst, col) in arr.iter_mut().zip(&key_slices[..N]) {
        *dst = col[row];
    }
    FixedKey(arr)
}

// -----------------------------------------------------------------------------
// Key Operations Trait & Implementations
// -----------------------------------------------------------------------------

pub(super) trait RadixKeyOps: Send + Sync + Copy + 'static {
    type Key: Eq + Hash + Clone + Send + Sync + Debug;

    fn new(n_keys: usize) -> Self;
    fn n_keys(&self) -> usize;
    fn compute_hash(&self, key_slices: &[&[i64]], row: usize) -> u64;
    fn extract_key(&self, key_slices: &[&[i64]], row: usize) -> Self::Key;
    fn push_flat(keys_flat: &mut Vec<i64>, key: &Self::Key);
}

#[derive(Clone, Copy)]
pub(super) struct FixedKeyOps<const N: usize>;

impl<const N: usize> RadixKeyOps for FixedKeyOps<N> {
    type Key = FixedKey<N>;

    #[inline(always)]
    fn new(_: usize) -> Self {
        Self
    }

    #[inline(always)]
    fn n_keys(&self) -> usize {
        N
    }

    #[inline(always)]
    fn compute_hash(&self, key_slices: &[&[i64]], row: usize) -> u64 {
        compute_hash_fixed::<N>(key_slices, row)
    }

    #[inline(always)]
    fn extract_key(&self, key_slices: &[&[i64]], row: usize) -> Self::Key {
        extract_key_fixed::<N>(key_slices, row)
    }

    #[inline(always)]
    fn push_flat(keys_flat: &mut Vec<i64>, key: &Self::Key) {
        keys_flat.extend_from_slice(&key.0);
    }
}

#[derive(Clone, Copy)]
pub(super) struct CompositeKeyOps {
    n_keys: usize,
}

impl RadixKeyOps for CompositeKeyOps {
    type Key = CompositeKey;

    #[inline(always)]
    fn new(n: usize) -> Self {
        Self { n_keys: n }
    }

    #[inline(always)]
    fn n_keys(&self) -> usize {
        self.n_keys
    }

    #[inline(always)]
    fn compute_hash(&self, key_slices: &[&[i64]], row: usize) -> u64 {
        compute_hash(key_slices, row)
    }

    #[inline(always)]
    fn extract_key(&self, key_slices: &[&[i64]], row: usize) -> Self::Key {
        extract_key(key_slices, row)
    }

    #[inline(always)]
    fn push_flat(keys_flat: &mut Vec<i64>, key: &Self::Key) {
        keys_flat.extend(key.iter().copied());
    }
}

// -----------------------------------------------------------------------------
// Unified Engine
// -----------------------------------------------------------------------------
