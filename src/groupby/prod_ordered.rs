use ahash::AHashMap;
use pyo3::prelude::*;

use super::chunks::validate_firstseen_deterministic_inputs;
use super::order::FirstSeenRowIndex;
use super::result::GroupByResultF64;

pub(super) fn groupby_prod_f64_ordered_low<I: FirstSeenRowIndex>(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    let n_rows = validate_firstseen_deterministic_inputs::<f64, I>(keys, values)?;

    let mut gid_by_key: AHashMap<i64, usize> = AHashMap::default();
    let mut keys_by_gid: Vec<i64> = Vec::new();
    let mut products_by_gid: Vec<f64> = Vec::new();

    for row in 0..n_rows {
        let key = keys[row];
        let gid = match gid_by_key.get(&key).copied() {
            Some(existing_gid) => existing_gid,
            None => {
                let next_gid = keys_by_gid.len();
                gid_by_key.insert(key, next_gid);
                keys_by_gid.push(key);
                products_by_gid.push(1.0);
                next_gid
            }
        };

        let value = values[row];
        if !value.is_nan() {
            products_by_gid[gid] *= value;
        }
    }

    Ok(GroupByResultF64 {
        keys: keys_by_gid,
        values: products_by_gid,
    })
}
