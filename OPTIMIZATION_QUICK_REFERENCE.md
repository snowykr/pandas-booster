# Quick Reference: Optimization Roadmap for pandas-booster

## TL;DR: Key Findings

✅ **Variance computation is already correct** (Chan merging implemented)  
✅ **Hash-based aggregation with first-seen reordering already implemented**  
⚠️ **Memory layout can be optimized** (GFTR pattern)  
⚠️ **High-cardinality workloads need algorithm selection** (hash vs radix vs sort)

---

## Phase 1: Quick Wins (1-2 weeks) - HIGH PRIORITY

### 1. GFTR Pattern (1.2-1.5x speedup on unsorted groupby)
**File**: `src/groupby.rs`, lines 41-87

**Change**: Gather keys and values together during reordering (not separately)

```rust
// Before: separate reordering (cache misses)
for &g in &perm {
    sorted_keys.push(keys[g]);
    sorted_values.push(values[g]);
}

// After: sequential gather (GFTR pattern)
let mut sorted_pairs = Vec::with_capacity(keys.len());
for &g in &perm {
    sorted_pairs.push((keys[g], values[g]));
}
```

**Why**: Reduces memory bandwidth by gathering both fields together.

---

### 2. Zero-Alloc Hash Table Pre-allocation (1.1-1.3x speedup on high-cardinality)
**File**: `src/groupby.rs`, hash table initialization

**Change**: Pre-allocate hash table with capacity estimate

```rust
// Before: dynamic allocation
let mut local_agg: AHashMap<i64, VarAggF64> = AHashMap::new();

// After: pre-allocated
let estimated_cardinality = estimate_cardinality(keys, 10_000);
let capacity = (estimated_cardinality as f64 * 1.5) as usize;
let mut local_agg: AHashMap<i64, VarAggF64> = AHashMap::with_capacity(capacity);
```

**Why**: Eliminates resizing stalls during aggregation.

---

### 3. Document Welford/Chan Merging (maintainability)
**File**: `src/aggregation.rs`, lines 86-150

**Add**: Code comments explaining Chan merging formula and numerical stability

**Why**: Future developers understand the algorithm and can optimize safely.

---

## Phase 2: Medium-Term (2-4 weeks) - MEDIUM PRIORITY

### 4. Cardinality-Aware Algorithm Selection (1.5-2x speedup on high-cardinality)
**File**: `src/groupby.rs`, new function

**Add**:
```rust
fn estimate_cardinality(keys: &[i64], sample_size: usize) -> usize { ... }

fn select_aggregation_strategy(cardinality: usize) -> Strategy {
    if cardinality < 1_000_000 {
        Strategy::HashBased  // Current
    } else if cardinality < 10_000_000 {
        Strategy::PartitionBased  // Radix partitioning
    } else {
        Strategy::SortBased  // For sort=True
    }
}
```

**Why**: Different algorithms dominate at different cardinalities.

---

## Phase 3: Advanced (4+ weeks) - LOW PRIORITY

### 5. Multi-Pass Aggregation for Cache Efficiency (1.8-2.5x speedup on >100M groups)
**File**: `src/groupby.rs`, new aggregation path

**Add**: Multi-pass aggregation that fits partial aggregates in L2 cache

**Why**: Cache misses dominate for very high cardinality.

---

## Decision Tree: When to Use Each Optimization

```
IF working on unsorted groupby (sort=False):
  → Implement GFTR pattern (Phase 1)
  
IF working on high-cardinality workloads (>1M groups):
  → Add zero-alloc pre-allocation (Phase 1)
  → Add cardinality-aware selection (Phase 2)
  
IF working on very high cardinality (>100M groups):
  → Consider multi-pass aggregation (Phase 3)
```

---

## Performance Prediction

| Cardinality | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|-------------|---------|---------------|---------------|---------------|
| <1M | 1.0x | 1.2x | 1.2x | 1.2x |
| 1-10M | 1.0x | 1.2x | 1.8x | 1.8x |
| >10M | 1.0x | 1.2x | 2.0x | 3.5x |

---

## Testing Strategy

1. **Phase 1**: Benchmark unsorted groupby on 5M rows, 1K-100K groups
2. **Phase 2**: Benchmark high-cardinality (5M rows, 5M groups)
3. **Phase 3**: Benchmark very high cardinality (100M rows, 100M groups)

---

## References

- **Welford/Chan**: Chan et al. (1983), "Algorithms for computing the sample variance"
- **GFTR Pattern**: Wu et al. (2025), "Efficiently Processing Joins and Grouped Aggregations on GPUs"
- **Hash vs Radix**: Xue & Marcus (2025), "Global Hash Tables Strike Back!"
- **Cache Effects**: Mueller et al. (2015), "Sorting vs. hashing revisited"

See `OPTIMIZATION_SYNTHESIS.md` for full details.

---

**Status**: Ready for implementation  
**Last Updated**: April 19, 2026
