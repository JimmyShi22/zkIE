//! Small shared helpers kept in the compose module: the piecewise rsqrt table
//! index rule (with its coverage check) and the one-pass sequence
//! self-consistency check for next-token claims.

use zkie_core::common::field::Goldilocks;

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if 2 * r >= b {
        q + 1
    } else {
        q
    }
}

/// Number of entries in the piecewise rsqrt table, fixed by
/// `scripts/extract_rsqrt_piecewise.py`: 2^20 step-1 entries, followed by 2^22
/// coarse entries at step 2^8 and 2^21 coarse entries at step 2^16.
pub const RSQRT_TABLE_LEN: usize = (1 << 20) + (1 << 22) + (1 << 21);

/// Largest real variance the generated table covers:
/// `(FINE + COARSE * COARSE_STEP + (COARSE2 - 1) * COARSE_STEP2) / INDEX_SCALE`.
pub const RSQRT_MAX_VARIANCE: f64 = 8454204.0;

/// Check a loaded rsqrt table covers the indices about to be read from it.
///
/// Deliberately a bounds check and not `len() == RSQRT_TABLE_LEN`: the ops work
/// with any table large enough for the data, and the unit tests legitimately use
/// smaller ones. What must never happen is reading past the end.
///
/// Without this a short table fails as a bare `index out of bounds` from inside
/// a forward pass, minutes into a run and a long way from the cause. It is a
/// live mistake rather than a hypothetical one: two generators exist, and the
/// superseded `scripts/extract_rsqrt_table.py` emits 2^19 uniform entries, so
/// running the wrong one produces a file that looks perfectly normal.
pub fn check_rsqrt_table(table: &[Goldilocks], max_index: u32) {
    assert!(
        (max_index as usize) < table.len(),
        "rsqrt table has {} entries but the forward pass needs index {}. \
         Regenerate with scripts/extract_rsqrt_piecewise.py ({} entries, covering \
         variance up to {:.6}) - NOT extract_rsqrt_table.py, which is superseded and \
         emits 2^19 uniform entries.",
        table.len(),
        max_index,
        RSQRT_TABLE_LEN,
        RSQRT_MAX_VARIANCE
    );
}

/// Piecewise rsqrt table index: fine step 1 below 2^20, coarse step 2^8 up to
/// the medium end, coarse step 2^16 beyond it.
pub fn rsqrt_index(var: i64) -> u32 {
    const FINE: i64 = 1 << 20;
    const COARSE: i64 = 1 << 22;
    const COARSE_END: i64 = FINE + COARSE * (1 << 8);
    let raw = round_div(var, 1 << 18);
    if raw < FINE {
        raw.max(0) as u32
    } else if raw < COARSE_END {
        (FINE + ((raw - FINE) >> 8)) as u32
    } else {
        (FINE + COARSE + ((raw - COARSE_END) >> 16)) as u32
    }
}

/// Verify the one-pass sequence self-consistency: the claimed sequence starts
/// with the user prompt, and every generated token equals the argmax prediction
/// of the previous position (teacher forcing). `tokens` is the full claimed
/// sequence [prompt + response], `argmax_t` is the model's argmax per position.
pub fn verify_sequence_shift(tokens: &[u32], argmax_t: &[u32], prompt: &[u32]) -> bool {
    if prompt.len() > tokens.len() || argmax_t.len() != tokens.len() {
        return false;
    }
    if tokens[..prompt.len()] != *prompt {
        return false;
    }
    for i in prompt.len().saturating_sub(1)..tokens.len().saturating_sub(1) {
        if argmax_t[i] != tokens[i + 1] {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::verify_sequence_shift;

    #[test]
    fn verify_sequence_shift_roundtrip() {
        // tokens = [prompt=10,20, generated=30,40,50]
        let tokens: Vec<u32> = vec![10, 20, 30, 40, 50];
        let prompt: Vec<u32> = vec![10, 20];
        // argmax at i predicts token i+1 for i in [prompt.len()-1, m-1)
        let argmax_t: Vec<u32> = vec![0, 30, 40, 50, 0];
        assert!(verify_sequence_shift(&tokens, &argmax_t, &prompt));
        // tamper a shift edge -> reject
        let mut bad = argmax_t.clone();
        bad[1] = 999;
        assert!(!verify_sequence_shift(&tokens, &bad, &prompt));
        // tamper the prefix -> reject
        let mut bad_prompt = prompt.clone();
        bad_prompt[0] = 999;
        assert!(!verify_sequence_shift(&tokens, &argmax_t, &bad_prompt));
    }
}
