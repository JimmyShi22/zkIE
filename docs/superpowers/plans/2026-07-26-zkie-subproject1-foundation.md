# zkIE Sub-Project 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build `zkie-core`'s foundation layer (fixed-point numbers, a generic range-check
gate, the ELTWISE Add/Mul chips, the ISA/Chip scaffolding, and Tensor) and prove it works
with a real (non-mocked) KZG setup→prove→verify roundtrip.

**Architecture:** Pure-Rust host-side fixed-point types (`I18`) feed witnesses into Halo2
circuits built from a small library of composable gates: a generic N-bit bit-decomposition
range check, an Add gate, and a Mul gate with a quotient/remainder rescale step. All chips
follow the standard halo2 `Config`/`Circuit` pattern.

**Tech Stack:** Rust, `halo2_proofs` (PSE fork, git tag `v0.4.0`), `halo2curves::bn256`
(re-exported via `halo2_proofs::halo2curves`), `rand_core`.

## Global Constraints

- `halo2_proofs` dependency: `{ git = "https://github.com/privacy-scaling-explorations/halo2.git", tag = "v0.4.0" }` — verified to compile and pass tests in this environment via a throwaway spike (`.spike-test/`, gitignored, deleted once this plan's own tests passed).
- Use `halo2_proofs::halo2curves::bn256::{Fr, G1Affine, Bn256}` (the re-export), not a separate direct `halo2curves` dependency — avoids version-mismatch risk.
- **Cargo/build commands must be run from inside `/Users/jimmyshi/code/zkie/...`, never from `/tmp` or the scratchpad** — local security software SIGKILLs build scripts (`build.rs`) run from temp paths. See root `CLAUDE.md`.
- `Circuit::synthesize` returns `halo2_proofs::plonk::ErrorFront` in this version (not the plain `Error` alias used in older halo2 tutorials).
- Transcript types need `use halo2_proofs::transcript::{TranscriptReadBuffer, TranscriptWriterBuffer};` in scope for `Blake2bRead`/`Blake2bWrite::init`.
- `create_proof`/`verify_proof` instances parameter type is `&[Vec<Vec<Fr>>]` (owned `Vec`s), not slices of references.
- `SingleStrategy::new` and `verify_proof` take `&ParamsVerifierKZG`, obtained via `params.verifier_params()` — pass it by reference (`&verifier_params`), not the original `&params`.
- Fixed-point convention for this sub-project: **I18** = signed integer scaled by `10^18`, stored host-side as `i64`, representable range exactly `i64::MIN..=i64::MAX` (a 64-bit signed range check). This is intentionally small (max representable magnitude ≈ 9.22) — sufficient for foundation validation; widening the bit-width is future work once real TimesFM activation ranges are known (see spec's "Open Items").
- Remainder-bound convention: floor/Euclidean division — `r` in `a*b = q*SCALE_18 + r` always satisfies `0 <= r < SCALE_18`, via Rust's `div_euclid`/`rem_euclid` host-side, and the "double decomposition" gadget circuit-side (Task 4).
- No lookup arguments in this sub-project — everything is custom gates + bit-decomposition, to avoid depending on an unverified external range-check crate (see spec §3.1, §5).

---

### Task 1: Cargo workspace + `zkie-core` skeleton

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/zkie-core/Cargo.toml`
- Create: `crates/zkie-core/src/lib.rs`

**Interfaces:**
- Produces: an empty `zkie-core` lib crate that `cargo test` can run against.

- [x] **Step 1: Create the workspace manifest**

`Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["crates/zkie-core"]
```

- [x] **Step 2: Create the `zkie-core` crate manifest**

`crates/zkie-core/Cargo.toml`:
```toml
[package]
name = "zkie-core"
version = "0.1.0"
edition = "2021"

[dependencies]
halo2_proofs = { git = "https://github.com/privacy-scaling-explorations/halo2.git", tag = "v0.4.0" }
rand_core = { version = "0.6", features = ["getrandom"] }
```

- [x] **Step 3: Create an empty lib with a placeholder test**

`crates/zkie-core/src/lib.rs`:
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {
        assert_eq!(2 + 2, 4);
    }
}
```

- [x] **Step 4: Run test to verify the workspace builds**

Run (from `/Users/jimmyshi/code/zkie`): `cargo test -p zkie-core`
Expected: `test tests::crate_compiles ... ok` (first run downloads/compiles `halo2_proofs` and its dependency tree — can take a few minutes).

- [x] **Step 5: Commit**

```bash
git add Cargo.toml crates/zkie-core/Cargo.toml crates/zkie-core/src/lib.rs
git commit -m "chore: scaffold zkie-core cargo workspace"
```

---

### Task 2: `fixed_point.rs` — I18 host type

**Files:**
- Create: `crates/zkie-core/src/fixed_point.rs`
- Modify: `crates/zkie-core/src/lib.rs` (add `pub mod fixed_point;`)

**Interfaces:**
- Produces:
  - `pub const SCALE_18: i128 = 1_000_000_000_000_000_000;`
  - `pub struct I18(i64);` with `I18::from_f64(f64) -> Result<I18, FixedPointError>`, `I18::to_f64(&self) -> f64`, `I18::raw(&self) -> i64`, `I18::from_raw(i64) -> I18`
  - `pub struct FixedPointError(String)` implementing `std::fmt::Display` + `std::error::Error`
  - `pub fn requantize_mul(a: I18, b: I18) -> Result<(I18, i128), FixedPointError>` returning `(quotient_as_I18, remainder)` where `remainder` satisfies `0 <= remainder < SCALE_18`
  - Later tasks (3+) consume `I18::raw()` (an `i64`) to build circuit witnesses, and `requantize_mul` to compute the expected `(q, r)` witness pair for the Mul chip.

- [x] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f64_round_trips_small_values() {
        let x = I18::from_f64(2.5).unwrap();
        assert!((x.to_f64() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn from_f64_round_trips_negative_values() {
        let x = I18::from_f64(-3.25).unwrap();
        assert!((x.to_f64() - (-3.25)).abs() < 1e-9);
    }

    #[test]
    fn from_f64_rejects_out_of_range() {
        // i64::MAX / 1e18 ~= 9.22, so 100.0 must not fit.
        assert!(I18::from_f64(100.0).is_err());
    }

    #[test]
    fn from_f64_accepts_boundary_value() {
        let max_representable = (i64::MAX as f64) / (SCALE_18 as f64);
        assert!(I18::from_f64(max_representable * 0.999).is_ok());
    }

    #[test]
    fn requantize_mul_positive_times_positive() {
        let a = I18::from_f64(2.0).unwrap();
        let b = I18::from_f64(3.0).unwrap();
        let (q, r) = requantize_mul(a, b).unwrap();
        assert!((q.to_f64() - 6.0).abs() < 1e-9);
        assert_eq!(r, 0);
    }

    #[test]
    fn requantize_mul_rounds_toward_negative_infinity_with_nonneg_remainder() {
        // 0.1 * 0.1 = 0.01 exactly in real arithmetic, but in I18 fixed point
        // 0.1 is not exactly representable, so check the remainder invariant
        // instead of an exact value.
        let a = I18::from_f64(0.1).unwrap();
        let b = I18::from_f64(0.1).unwrap();
        let (_, r) = requantize_mul(a, b).unwrap();
        assert!(r >= 0 && r < SCALE_18);
    }

    #[test]
    fn requantize_mul_negative_times_positive_remainder_nonnegative() {
        let a = I18::from_f64(-2.5).unwrap();
        let b = I18::from_f64(2.0).unwrap();
        let (q, r) = requantize_mul(a, b).unwrap();
        assert!((q.to_f64() - (-5.0)).abs() < 1e-9);
        assert!(r >= 0 && r < SCALE_18);
    }
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cargo test -p zkie-core fixed_point`
Expected: FAIL with "cannot find function/struct" compile errors (module doesn't exist yet).

- [x] **Step 3: Implement**

```rust
use std::fmt;

pub const SCALE_18: i128 = 1_000_000_000_000_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedPointError(pub String);

impl fmt::Display for FixedPointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fixed point error: {}", self.0)
    }
}

impl std::error::Error for FixedPointError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I18(i64);

impl I18 {
    pub fn from_raw(raw: i64) -> Self {
        I18(raw)
    }

    pub fn raw(&self) -> i64 {
        self.0
    }

    pub fn from_f64(value: f64) -> Result<Self, FixedPointError> {
        let scaled = value * (SCALE_18 as f64);
        if !scaled.is_finite() {
            return Err(FixedPointError(format!("{value} is not finite")));
        }
        let rounded = scaled.round();
        if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
            return Err(FixedPointError(format!(
                "{value} does not fit in the I18 representable range"
            )));
        }
        Ok(I18(rounded as i64))
    }

    pub fn to_f64(&self) -> f64 {
        (self.0 as f64) / (SCALE_18 as f64)
    }
}

/// Computes `a * b` in I18 fixed point, returning the requantized I18 result
/// (`quotient`) and the Euclidean remainder (`0 <= remainder < SCALE_18`).
/// Returns an error if the requantized quotient overflows I18's range.
pub fn requantize_mul(a: I18, b: I18) -> Result<(I18, i128), FixedPointError> {
    let raw_product: i128 = (a.raw() as i128) * (b.raw() as i128);
    let quotient = raw_product.div_euclid(SCALE_18);
    let remainder = raw_product.rem_euclid(SCALE_18);
    if quotient < i64::MIN as i128 || quotient > i64::MAX as i128 {
        return Err(FixedPointError(format!(
            "product of {} and {} overflows I18 range",
            a.to_f64(),
            b.to_f64()
        )));
    }
    Ok((I18(quotient as i64), remainder))
}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cargo test -p zkie-core fixed_point`
Expected: all tests `ok`.

- [x] **Step 5: Add the module to `lib.rs`**

```rust
pub mod fixed_point;
```

- [x] **Step 6: Commit**

```bash
git add crates/zkie-core/src/fixed_point.rs crates/zkie-core/src/lib.rs
git commit -m "feat: add I18 fixed-point number system"
```

---

### Task 3: `field_convert.rs` — `i64`/`i128` ↔ `Fr` helpers

**Files:**
- Create: `crates/zkie-core/src/field_convert.rs`
- Modify: `crates/zkie-core/src/lib.rs` (add `pub mod field_convert;`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `pub fn i64_to_fr(v: i64) -> Fr` and `pub fn i128_to_fr(v: i128) -> Fr` (used by Task 4/5 to build circuit witnesses from `I18::raw()` / remainder values). `Fr` re-exported as `pub use halo2_proofs::halo2curves::bn256::Fr;` from this module for convenience.

- [x] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_i64_round_trips_via_u64() {
        let fr = i64_to_fr(42);
        assert_eq!(fr, Fr::from(42u64));
    }

    #[test]
    fn negative_i64_is_additive_inverse_in_field() {
        let fr = i64_to_fr(-42);
        assert_eq!(fr, -Fr::from(42u64));
    }

    #[test]
    fn zero_maps_to_field_zero() {
        assert_eq!(i64_to_fr(0), Fr::zero());
    }

    #[test]
    fn negative_i128_is_additive_inverse_in_field() {
        let fr = i128_to_fr(-123456789012345i128);
        assert_eq!(fr, -Fr::from_u128(123456789012345u128));
    }
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cargo test -p zkie-core field_convert`
Expected: FAIL — module/functions don't exist.

- [x] **Step 3: Implement**

```rust
pub use halo2_proofs::halo2curves::bn256::Fr;
use halo2_proofs::halo2curves::ff::PrimeField;

pub fn i64_to_fr(v: i64) -> Fr {
    if v >= 0 {
        Fr::from(v as u64)
    } else {
        -Fr::from((-(v as i128)) as u64)
    }
}

pub fn i128_to_fr(v: i128) -> Fr {
    if v >= 0 {
        Fr::from_u128(v as u128)
    } else {
        -Fr::from_u128((-v) as u128)
    }
}
```

If `Fr::from_u128` is not available on this `PrimeField`/`Field` impl, the compiler error
from Step 4 will name the correct trait/method — adjust the two branches above to use
whatever the compiler suggests (e.g. decomposing into two `u64` limbs) and re-run. Do not
guess further method names beyond what the compiler reports.

- [x] **Step 4: Run tests to verify they pass**

Run: `cargo test -p zkie-core field_convert`
Expected: all tests `ok`. If `from_u128`/`PrimeField` import path errors surface, fix per
the compiler's suggestion and re-run this step before moving on.

- [x] **Step 5: Add the module to `lib.rs`**

```rust
pub mod field_convert;
```

- [x] **Step 6: Commit**

```bash
git add crates/zkie-core/src/field_convert.rs crates/zkie-core/src/lib.rs
git commit -m "feat: add i64/i128 to Fr conversion helpers"
```

---

### Task 4: `chips/range_check.rs` — generic bit-decomposition range check

**Files:**
- Create: `crates/zkie-core/src/chips/mod.rs`
- Create: `crates/zkie-core/src/chips/range_check.rs`
- Modify: `crates/zkie-core/src/lib.rs` (add `pub mod chips;`)

**Interfaces:**
- Consumes: `Fr` from `field_convert`.
- Produces:
  - `pub struct RangeCheckConfig { pub value: Column<Advice>, pub bits: Column<Advice>, pub s_bit: Selector, pub s_sum: Selector, pub n_bits: usize }`
  - `pub struct RangeCheckChip { config: RangeCheckConfig }` with:
    - `RangeCheckChip::configure(meta: &mut ConstraintSystem<Fr>, value: Column<Advice>, bits: Column<Advice>, n_bits: usize) -> RangeCheckConfig`
    - `RangeCheckChip::construct(config: RangeCheckConfig) -> Self`
    - `fn assign(&self, layouter: impl Layouter<Fr>, value: Value<Fr>, raw_value: Value<i128>) -> Result<AssignedCell<Fr, Fr>, ErrorFront>` — assigns `value` into the `value` column, decomposes `raw_value` (an unsigned, already-shifted-if-signed magnitude, known only in the witness-generation context) into `n_bits` boolean cells in `bits` (one row per bit, using `n_bits` rows total within its own region), and enforces (via `s_sum` at the last bit row, referencing all prior rows through `Rotation`) that the running sum of `bit_i * 2^i` equals `value`. Returns the assigned `value` cell so callers can `copy_advice`/`constrain_instance` it elsewhere.
  - Later tasks (5) call `RangeCheckChip::configure`/`assign` twice per Eltwise chip (once for each range-checked quantity), and use the "shift by `2^63`" trick for signed 64-bit values, and the "double decomposition" trick (two `RangeCheckChip` instances: one on `r`, one on `SCALE_18 - 1 - r`, both with `n_bits = 60`) for the Mul remainder bound.

- [x] **Step 1: Write the failing tests (in `range_check.rs`, using `MockProver`)**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::circuit::{SimpleFloorPlanner, Value};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};
    use crate::field_convert::Fr;

    #[derive(Clone)]
    struct TestConfig {
        range: RangeCheckConfig,
    }

    struct TestCircuit {
        value: Value<Fr>,
        raw_value: Value<i128>,
        n_bits: usize,
    }

    impl Circuit<Fr> for TestCircuit {
        type Config = TestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            TestCircuit { value: Value::unknown(), raw_value: Value::unknown(), n_bits: self.n_bits }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let value = meta.advice_column();
            let bits = meta.advice_column();
            meta.enable_equality(value);
            let range = RangeCheckChip::configure(meta, value, bits, 8);
            TestConfig { range }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = RangeCheckChip::construct(config.range);
            chip.assign(layouter, self.value, self.raw_value)?;
            Ok(())
        }
    }

    #[test]
    fn value_within_8_bit_range_is_satisfied() {
        let circuit = TestCircuit { value: Value::known(Fr::from(200u64)), raw_value: Value::known(200i128), n_bits: 8 };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn value_equal_to_zero_is_satisfied() {
        let circuit = TestCircuit { value: Value::known(Fr::zero()), raw_value: Value::known(0i128), n_bits: 8 };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn value_at_max_of_range_is_satisfied() {
        let circuit = TestCircuit { value: Value::known(Fr::from(255u64)), raw_value: Value::known(255i128), n_bits: 8 };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn value_exceeding_8_bit_range_is_rejected() {
        // 256 does not fit in 8 bits: the running-sum decomposition of the
        // claimed bits cannot equal 256 if all 8 bits are constrained boolean,
        // so the assigned `value` witness (256) will mismatch the sum (<=255).
        let circuit = TestCircuit { value: Value::known(Fr::from(256u64)), raw_value: Value::known(256i128), n_bits: 8 };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cargo test -p zkie-core range_check`
Expected: FAIL — `RangeCheckChip`/`RangeCheckConfig` don't exist yet.

- [x] **Step 3: Implement**

```rust
use crate::field_convert::Fr;
use halo2_proofs::circuit::{AssignedCell, Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;

#[derive(Clone, Debug)]
pub struct RangeCheckConfig {
    pub value: Column<Advice>,
    pub bits: Column<Advice>,
    pub s_bit: Selector,
    pub s_sum: Selector,
    pub n_bits: usize,
}

pub struct RangeCheckChip {
    config: RangeCheckConfig,
}

impl RangeCheckChip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        value: Column<Advice>,
        bits: Column<Advice>,
        n_bits: usize,
    ) -> RangeCheckConfig {
        meta.enable_equality(bits);

        let s_bit = meta.selector();
        meta.create_gate("bit is boolean", |meta| {
            let bit = meta.query_advice(bits, Rotation::cur());
            let s_bit = meta.query_selector(s_bit);
            let one = Expression::Constant(Fr::one());
            vec![s_bit * bit.clone() * (one - bit)]
        });

        let s_sum = meta.selector();
        meta.create_gate("running sum equals value", |meta| {
            let value = meta.query_advice(value, Rotation::cur());
            let s_sum = meta.query_selector(s_sum);
            let mut sum = Expression::Constant(Fr::zero());
            let mut coeff = Fr::one();
            for i in 0..n_bits {
                let rotation = Rotation(-(i as i32));
                let bit = meta.query_advice(bits, rotation);
                sum = sum + bit * Expression::Constant(coeff);
                coeff = coeff.double();
            }
            vec![s_sum * (sum - value)]
        });

        RangeCheckConfig { value, bits, s_bit, s_sum, n_bits }
    }

    pub fn construct(config: RangeCheckConfig) -> Self {
        RangeCheckChip { config }
    }

    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        value: Value<Fr>,
        raw_value: Value<i128>,
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        let n_bits = self.config.n_bits;
        layouter.assign_region(
            || "range check",
            |mut region| {
                for i in 0..n_bits {
                    self.config.s_bit.enable(&mut region, i)?;
                    let bit_value = raw_value.map(|v| ((v >> i) & 1) as u64);
                    region.assign_advice(
                        || format!("bit {i}"),
                        self.config.bits,
                        i,
                        || bit_value.map(Fr::from),
                    )?;
                }
                self.config.s_sum.enable(&mut region, n_bits - 1)?;
                region.assign_advice(|| "value", self.config.value, n_bits - 1, || value)
            },
        )
    }
}
```

Note: the running-sum gate at row `n_bits - 1` references rows `n_bits-1, n_bits-2, ..., 0`
via negative `Rotation`s — this only works because all `n_bits` bit cells and the `value`
cell for one `assign` call live in the same region at consecutive rows `0..n_bits`, with
`value` placed at the last row. If Step 4 reports a rotation/region-shape error, adjust by
placing `value` at row `0` and bits at rows `1..=n_bits` with positive rotations instead —
follow the compiler's actual error rather than guessing further.

- [x] **Step 4: Run tests to verify they pass**

Run: `cargo test -p zkie-core range_check`
Expected: all 4 tests `ok`, including the negative case failing verification.

- [x] **Step 5: Wire up `chips/mod.rs` and `lib.rs`**

`crates/zkie-core/src/chips/mod.rs`:
```rust
pub mod range_check;
```

`lib.rs`: add `pub mod chips;`

- [x] **Step 6: Commit**

```bash
git add crates/zkie-core/src/chips crates/zkie-core/src/lib.rs
git commit -m "feat: add generic bit-decomposition RangeCheckChip"
```

---

### Task 5: `chips/eltwise.rs` — Add and Mul chips

**Files:**
- Create: `crates/zkie-core/src/chips/eltwise.rs`
- Modify: `crates/zkie-core/src/chips/mod.rs` (add `pub mod eltwise;`)

**Interfaces:**
- Consumes: `RangeCheckChip`/`RangeCheckConfig` from Task 4, `Fr`/`i64_to_fr`/`i128_to_fr` from Task 3, `I18`/`requantize_mul`/`SCALE_18` from Task 2.
- Produces:
  - `pub struct EltwiseAddConfig { a: Column<Advice>, b: Column<Advice>, c: Column<Advice>, s_add: Selector, range_a: RangeCheckConfig, range_b: RangeCheckConfig, range_c: RangeCheckConfig }`
  - `pub struct EltwiseAddChip { config: EltwiseAddConfig }` with `configure(meta, a, b, c, bits_col) -> EltwiseAddConfig`, `construct(config) -> Self`, `fn assign(&self, layouter, a: I18, b: I18) -> Result<(), ErrorFront>` (assigns `a`, `b`, computed `c = a+b`, and range-checks all three as signed 64-bit values via the `2^63`-shift trick).
  - `pub struct EltwiseMulConfig { a: Column<Advice>, b: Column<Advice>, q: Column<Advice>, r: Column<Advice>, s_mul: Selector, range_q: RangeCheckConfig, range_r: RangeCheckConfig, range_r_slack: RangeCheckConfig }`
  - `pub struct EltwiseMulChip { config: EltwiseMulConfig }` with analogous `configure`/`construct`/`assign(&self, layouter, a: I18, b: I18) -> Result<(), ErrorFront>` implementing the quotient/remainder rescale gate `a*b - q*SCALE_18 - r = 0`, range-checking `q` as signed 64-bit and `r`/`SCALE_18-1-r` each as unsigned 60-bit (the double-decomposition bound).
  - Later tasks (7) instantiate these chips inside a top-level `Circuit` impl for the KZG roundtrip test.

- [x] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::I18;
    use halo2_proofs::circuit::SimpleFloorPlanner;
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    #[derive(Clone)]
    struct AddTestConfig {
        add: EltwiseAddConfig,
    }

    struct AddTestCircuit {
        a: I18,
        b: I18,
    }

    impl Circuit<Fr> for AddTestCircuit {
        type Config = AddTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            AddTestCircuit { a: I18::from_raw(0), b: I18::from_raw(0) }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let c = meta.advice_column();
            let bits = meta.advice_column();
            meta.enable_equality(a);
            meta.enable_equality(b);
            meta.enable_equality(c);
            AddTestConfig { add: EltwiseAddChip::configure(meta, a, b, c, bits) }
        }

        fn synthesize(&self, config: Self::Config, layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
            let chip = EltwiseAddChip::construct(config.add);
            chip.assign(layouter, self.a, self.b)
        }
    }

    #[test]
    fn add_positive_plus_positive_satisfied() {
        let circuit = AddTestCircuit { a: I18::from_f64(2.0).unwrap(), b: I18::from_f64(3.5).unwrap() };
        let prover = MockProver::run(8, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn add_negative_plus_positive_satisfied() {
        let circuit = AddTestCircuit { a: I18::from_f64(-2.0).unwrap(), b: I18::from_f64(3.5).unwrap() };
        let prover = MockProver::run(8, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[derive(Clone)]
    struct MulTestConfig {
        mul: EltwiseMulConfig,
    }

    struct MulTestCircuit {
        a: I18,
        b: I18,
    }

    impl Circuit<Fr> for MulTestCircuit {
        type Config = MulTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            MulTestCircuit { a: I18::from_raw(0), b: I18::from_raw(0) }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let q = meta.advice_column();
            let r = meta.advice_column();
            let bits = meta.advice_column();
            meta.enable_equality(a);
            meta.enable_equality(b);
            meta.enable_equality(q);
            meta.enable_equality(r);
            MulTestConfig { mul: EltwiseMulChip::configure(meta, a, b, q, r, bits) }
        }

        fn synthesize(&self, config: Self::Config, layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
            let chip = EltwiseMulChip::construct(config.mul);
            chip.assign(layouter, self.a, self.b)
        }
    }

    #[test]
    fn mul_positive_times_positive_satisfied() {
        let circuit = MulTestCircuit { a: I18::from_f64(2.0).unwrap(), b: I18::from_f64(3.0).unwrap() };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn mul_negative_times_positive_satisfied() {
        let circuit = MulTestCircuit { a: I18::from_f64(-2.5).unwrap(), b: I18::from_f64(2.0).unwrap() };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cargo test -p zkie-core eltwise`
Expected: FAIL — chips don't exist yet.

- [x] **Step 3: Implement**

```rust
use crate::chips::range_check::{RangeCheckChip, RangeCheckConfig};
use crate::field_convert::{i128_to_fr, i64_to_fr, Fr};
use crate::fixed_point::{requantize_mul, I18, SCALE_18};
use halo2_proofs::circuit::{Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;

const SIGNED_SHIFT: i128 = 1i128 << 63;
const REMAINDER_BITS: usize = 60; // 2^60 > SCALE_18 - 1, see plan Task 5 notes.

fn shifted_i64_witness(v: i64) -> (Value<Fr>, Value<i128>) {
    let shifted = (v as i128) + SIGNED_SHIFT;
    (Value::known(i128_to_fr(shifted)), Value::known(shifted))
}

#[derive(Clone, Debug)]
pub struct EltwiseAddConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    s_add: Selector,
    range_a: RangeCheckConfig,
    range_b: RangeCheckConfig,
    range_c: RangeCheckConfig,
}

pub struct EltwiseAddChip {
    config: EltwiseAddConfig,
}

impl EltwiseAddChip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        a: Column<Advice>,
        b: Column<Advice>,
        c: Column<Advice>,
        bits: Column<Advice>,
    ) -> EltwiseAddConfig {
        let s_add = meta.selector();
        meta.create_gate("add", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let s_add = meta.query_selector(s_add);
            vec![s_add * (a + b - c)]
        });

        let range_a = RangeCheckChip::configure(meta, a, bits, 64);
        let range_b = RangeCheckChip::configure(meta, b, bits, 64);
        let range_c = RangeCheckChip::configure(meta, c, bits, 64);

        EltwiseAddConfig { a, b, c, s_add, range_a, range_b, range_c }
    }

    pub fn construct(config: EltwiseAddConfig) -> Self {
        EltwiseAddChip { config }
    }

    pub fn assign(&self, mut layouter: impl Layouter<Fr>, a: I18, b: I18) -> Result<(), ErrorFront> {
        let c_raw = a.raw().checked_add(b.raw()).expect("I18 add overflow");

        layouter.assign_region(
            || "eltwise add",
            |mut region| {
                self.config.s_add.enable(&mut region, 0)?;
                region.assign_advice(|| "a", self.config.a, 0, || Value::known(i64_to_fr(a.raw())))?;
                region.assign_advice(|| "b", self.config.b, 0, || Value::known(i64_to_fr(b.raw())))?;
                region.assign_advice(|| "c", self.config.c, 0, || Value::known(i64_to_fr(c_raw)))?;
                Ok(())
            },
        )?;

        let (a_shift_fr, a_shift_raw) = shifted_i64_witness(a.raw());
        let range_a_chip = RangeCheckChip::construct(self.config.range_a.clone());
        range_a_chip.assign(layouter.namespace(|| "range a"), a_shift_fr, a_shift_raw)?;

        let (b_shift_fr, b_shift_raw) = shifted_i64_witness(b.raw());
        let range_b_chip = RangeCheckChip::construct(self.config.range_b.clone());
        range_b_chip.assign(layouter.namespace(|| "range b"), b_shift_fr, b_shift_raw)?;

        let (c_shift_fr, c_shift_raw) = shifted_i64_witness(c_raw);
        let range_c_chip = RangeCheckChip::construct(self.config.range_c.clone());
        range_c_chip.assign(layouter.namespace(|| "range c"), c_shift_fr, c_shift_raw)?;

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct EltwiseMulConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    q: Column<Advice>,
    r: Column<Advice>,
    s_mul: Selector,
    range_q: RangeCheckConfig,
    range_r: RangeCheckConfig,
    range_r_slack: RangeCheckConfig,
}

pub struct EltwiseMulChip {
    config: EltwiseMulConfig,
}

impl EltwiseMulChip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        a: Column<Advice>,
        b: Column<Advice>,
        q: Column<Advice>,
        r: Column<Advice>,
        bits: Column<Advice>,
    ) -> EltwiseMulConfig {
        let s_mul = meta.selector();
        meta.create_gate("mul rescale", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let q = meta.query_advice(q, Rotation::cur());
            let r = meta.query_advice(r, Rotation::cur());
            let s_mul = meta.query_selector(s_mul);
            let scale = Expression::Constant(i128_to_fr(SCALE_18));
            vec![s_mul * (a * b - q * scale - r)]
        });

        let range_q = RangeCheckChip::configure(meta, q, bits, 64);
        let range_r = RangeCheckChip::configure(meta, r, bits, REMAINDER_BITS);
        let range_r_slack = RangeCheckChip::configure(meta, r, bits, REMAINDER_BITS);

        EltwiseMulConfig { a, b, q, r, s_mul, range_q, range_r, range_r_slack }
    }

    pub fn construct(config: EltwiseMulConfig) -> Self {
        EltwiseMulChip { config }
    }

    pub fn assign(&self, mut layouter: impl Layouter<Fr>, a: I18, b: I18) -> Result<(), ErrorFront> {
        let (q, r) = requantize_mul(a, b).expect("I18 mul overflow");

        layouter.assign_region(
            || "eltwise mul",
            |mut region| {
                self.config.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "a", self.config.a, 0, || Value::known(i64_to_fr(a.raw())))?;
                region.assign_advice(|| "b", self.config.b, 0, || Value::known(i64_to_fr(b.raw())))?;
                region.assign_advice(|| "q", self.config.q, 0, || Value::known(i64_to_fr(q.raw())))?;
                region.assign_advice(|| "r", self.config.r, 0, || Value::known(i128_to_fr(r)))?;
                Ok(())
            },
        )?;

        let (q_shift_fr, q_shift_raw) = shifted_i64_witness(q.raw());
        let range_q_chip = RangeCheckChip::construct(self.config.range_q.clone());
        range_q_chip.assign(layouter.namespace(|| "range q"), q_shift_fr, q_shift_raw)?;

        let range_r_chip = RangeCheckChip::construct(self.config.range_r.clone());
        range_r_chip.assign(layouter.namespace(|| "range r"), Value::known(i128_to_fr(r)), Value::known(r))?;

        let slack = SCALE_18 - 1 - r;
        let range_r_slack_chip = RangeCheckChip::construct(self.config.range_r_slack.clone());
        range_r_slack_chip.assign(
            layouter.namespace(|| "range r slack"),
            Value::known(i128_to_fr(slack)),
            Value::known(slack),
        )?;

        Ok(())
    }
}
```

Note: `range_r` and `range_r_slack` are both configured against column `r`, which is only
valid if `RangeCheckChip::configure` doesn't assume exclusive ownership of that column's
selectors across the whole circuit — if Step 4 reports a "shared column/selector conflict"
error, give `range_r_slack` its own dedicated advice column (add a `slack` column parameter)
instead of reusing `r`, and witness `slack` explicitly. Follow the actual compiler/MockProver
error, not this note, as the source of truth.

- [x] **Step 4: Run tests to verify they pass**

Run: `cargo test -p zkie-core eltwise`
Expected: all 4 tests `ok`.

- [x] **Step 5: Add negative tests (bad witnesses must fail verification)**

```rust
    #[test]
    fn add_with_forged_sum_is_rejected() {
        // Reuse AddTestCircuit's shape but directly force a wrong `c` by
        // wrapping a circuit variant that assigns c = a + b + 1 instead.
        struct ForgedAddCircuit { a: I18, b: I18 }
        impl Circuit<Fr> for ForgedAddCircuit {
            type Config = AddTestConfig;
            type FloorPlanner = SimpleFloorPlanner;
            fn without_witnesses(&self) -> Self {
                ForgedAddCircuit { a: I18::from_raw(0), b: I18::from_raw(0) }
            }
            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                AddTestCircuit::configure(meta)
            }
            fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
                layouter.assign_region(
                    || "forged add",
                    |mut region| {
                        config.add_raw_gate_for_test().enable(&mut region, 0)?;
                        region.assign_advice(|| "a", config.add.a_col(), 0, || Value::known(i64_to_fr(self.a.raw())))?;
                        region.assign_advice(|| "b", config.add.b_col(), 0, || Value::known(i64_to_fr(self.b.raw())))?;
                        let forged_c = self.a.raw() + self.b.raw() + 1;
                        region.assign_advice(|| "c", config.add.c_col(), 0, || Value::known(i64_to_fr(forged_c)))
                    },
                )?;
                Ok(())
            }
        }
        // If EltwiseAddConfig's fields are private, expose the small test-only
        // accessors used above (`a_col`/`b_col`/`c_col`/`add_raw_gate_for_test`)
        // with `#[cfg(test)] pub(crate)` visibility, or — simpler — inline the
        // gate/column construction directly in this test rather than reusing
        // EltwiseAddChip::configure. Prefer the simpler inline approach if the
        // accessor plumbing feels awkward; the goal is just: assign a witness
        // that violates `a + b == c` and confirm MockProver rejects it.
    }
```

This step is intentionally left as a design note rather than fully worked code: the exact
shape of "assign a deliberately-wrong witness against the real gate" depends on which
fields end up `pub(crate)` after Step 3 is implemented for real. When implementing, prefer
the simplest version: build a small local circuit that calls `EltwiseAddChip::configure`
normally, then in `synthesize` assign `a`, `b`, and a forged `c != a+b` directly via
`region.assign_advice` (bypassing `EltwiseAddChip::assign`), and confirm
`MockProver::run(...).verify().is_err()`. Do the same for `EltwiseMulChip` with a forged
`q`/`r` pair that doesn't satisfy `a*b == q*SCALE_18+r`.

- [x] **Step 6: Run tests to verify the negative cases pass**

Run: `cargo test -p zkie-core eltwise`
Expected: all tests `ok`, including the new negative cases failing verification as expected.

- [x] **Step 7: Wire up module and commit**

`chips/mod.rs`: add `pub mod eltwise;`

```bash
git add crates/zkie-core/src/chips/eltwise.rs crates/zkie-core/src/chips/mod.rs
git commit -m "feat: add EltwiseAddChip and EltwiseMulChip"
```

---

### Task 6: `isa.rs`, `chip.rs`, `tensor.rs`

**Files:**
- Create: `crates/zkie-core/src/isa.rs`
- Create: `crates/zkie-core/src/chip.rs`
- Create: `crates/zkie-core/src/tensor.rs`
- Modify: `crates/zkie-core/src/lib.rs`

**Interfaces:**
- Consumes: nothing structurally new (this task defines the shared vocabulary; `EltwiseAddChip`/`EltwiseMulChip` from Task 5 are the only ones that currently implement `Chip`).
- Produces: `pub enum Instruction { DotGeneral { .. }, Softmax { .. }, Gelu, LayerNorm { .. }, Eltwise { op: EltwiseOp }, Reduce { .. }, EmbedLookup { .. }, PatchEmbed { .. } }` (fields per the parent design doc §3.1), `pub enum EltwiseOp { Add, Mul, Relu }`, `pub trait Chip { type Input; fn assign(&self, layouter: impl Layouter<Fr>, input: Self::Input) -> Result<(), ErrorFront>; }`, `pub struct Tensor<T> { pub shape: Vec<usize>, pub data: Vec<T> }` with `Tensor::new(shape, data) -> Result<Self, String>` (validates `data.len() == shape.iter().product()`).

- [x] **Step 1: Write the failing tests**

```rust
// isa.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eltwise_add_variant_constructs() {
        let instr = Instruction::Eltwise { op: EltwiseOp::Add };
        matches!(instr, Instruction::Eltwise { op: EltwiseOp::Add });
    }
}

// tensor.rs
#[cfg(test)]
mod tensor_tests {
    use super::*;

    #[test]
    fn new_accepts_matching_shape_and_data_len() {
        let t = Tensor::new(vec![2, 3], vec![0i64; 6]).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
    }

    #[test]
    fn new_rejects_mismatched_shape_and_data_len() {
        assert!(Tensor::new(vec![2, 3], vec![0i64; 5]).is_err());
    }
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cargo test -p zkie-core isa tensor`
Expected: FAIL — types don't exist yet.

- [x] **Step 3: Implement**

`isa.rs`:
```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EltwiseOp {
    Add,
    Mul,
    Relu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    DotGeneral { m: usize, n: usize, k: usize, batch_dims: Vec<usize>, trans_a: bool, trans_b: bool },
    Softmax { axis_dim: usize },
    Gelu,
    LayerNorm { dim: usize, epsilon_milli: u64 },
    Eltwise { op: EltwiseOp },
    Reduce { op: ReduceOp, axis: usize },
    EmbedLookup { table_size: usize, embed_dim: usize },
    PatchEmbed { patch_len: usize, embed_dim: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
}
```

`chip.rs`:
```rust
use crate::field_convert::Fr;
use halo2_proofs::circuit::Layouter;
use halo2_proofs::plonk::ErrorFront;

pub trait Chip {
    type Input;

    fn assign(&self, layouter: impl Layouter<Fr>, input: Self::Input) -> Result<(), ErrorFront>;
}
```

`tensor.rs`:
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> Tensor<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected
            ));
        }
        Ok(Tensor { shape, data })
    }
}
```

Note: `Chip::assign` here is a design-vocabulary trait, distinct from the concrete
`assign(&self, layouter, a: I18, b: I18)` methods already written directly on
`EltwiseAddChip`/`EltwiseMulChip` in Task 5 (which take two `I18` operands, not a single
generic `Input`). Retrofitting Task 5's chips to implement this trait (e.g. via an
`Input = (I18, I18)` associated type) is optional cleanup for this sub-project — do it only
if it's a trivial signature match; otherwise leave `Chip` as a documented-but-unused
placeholder for sub-project 2, where the remaining chips will implement it directly. Do not
force an awkward retrofit just to satisfy the trait.

- [x] **Step 4: Run tests to verify they pass**

Run: `cargo test -p zkie-core isa tensor`
Expected: all tests `ok`.

- [x] **Step 5: Wire up `lib.rs` and commit**

```rust
pub mod isa;
pub mod chip;
pub mod tensor;
```

```bash
git add crates/zkie-core/src/isa.rs crates/zkie-core/src/chip.rs crates/zkie-core/src/tensor.rs crates/zkie-core/src/lib.rs
git commit -m "feat: add ISA Instruction enum, Chip trait, and Tensor type"
```

---

### Task 7: Real KZG roundtrip integration test

**Files:**
- Create: `crates/zkie-core/tests/kzg_roundtrip.rs`

**Interfaces:**
- Consumes: `EltwiseAddChip`/`EltwiseAddConfig`, `EltwiseMulChip`/`EltwiseMulConfig` from Task 5, `I18` from Task 2, `Fr` from Task 3. Uses the exact KZG setup/keygen/prove/verify sequence already verified working in `.spike-test/` (see plan's Global Constraints for the API quirks already resolved there — `ErrorFront`, `TranscriptReadBuffer`/`TranscriptWriterBuffer` imports, `Vec<Vec<Fr>>` instances shape, `&verifier_params`).
- Produces: nothing consumed by later tasks — this is the sub-project's terminal deliverable.

- [x] **Step 1: Write the test file**

```rust
use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::dev::MockProver;
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ConstraintSystem, ErrorFront,
};
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
};
use halo2_proofs::halo2curves::bn256::{Bn256, G1Affine};
use rand_core::OsRng;
use zkie_core::chips::eltwise::{EltwiseAddChip, EltwiseAddConfig, EltwiseMulChip, EltwiseMulConfig};
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

#[derive(Clone)]
struct AddCircuitConfig {
    add: EltwiseAddConfig,
}

struct AddCircuit {
    a: I18,
    b: I18,
}

impl Circuit<Fr> for AddCircuit {
    type Config = AddCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        AddCircuit { a: I18::from_raw(0), b: I18::from_raw(0) }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let bits = meta.advice_column();
        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(c);
        AddCircuitConfig { add: EltwiseAddChip::configure(meta, a, b, c, bits) }
    }

    fn synthesize(&self, config: Self::Config, layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
        EltwiseAddChip::construct(config.add).assign(layouter, self.a, self.b)
    }
}

#[test]
fn eltwise_add_real_kzg_roundtrip() {
    let k = 10;
    let mut rng = OsRng;
    let circuit = AddCircuit { a: I18::from_f64(2.0).unwrap(), b: I18::from_f64(3.5).unwrap() };

    // Sanity check with MockProver first (fast) before paying for real KZG setup.
    MockProver::run(k, &circuit, vec![]).unwrap().assert_satisfied();

    let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk should not fail");

    let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        &[vec![]],
        &mut rng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
    let proof = transcript.finalize();

    let mut verifier_transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    let verifier_params = params.verifier_params();
    let strategy = SingleStrategy::new(&verifier_params);
    let result = verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<Bn256>, _, _, _>(
        &verifier_params,
        &vk,
        strategy,
        &[vec![]],
        &mut verifier_transcript,
    );
    assert!(result.is_ok(), "eltwise add proof failed to verify: {:?}", result);
}

#[test]
fn eltwise_add_tampered_proof_fails_verification() {
    let k = 10;
    let mut rng = OsRng;
    let circuit = AddCircuit { a: I18::from_f64(2.0).unwrap(), b: I18::from_f64(3.5).unwrap() };

    let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk should not fail");

    let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        &[vec![]],
        &mut rng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
    let mut proof = transcript.finalize();
    // Flip a byte in the middle of the proof to simulate tampering.
    let mid = proof.len() / 2;
    proof[mid] ^= 0xFF;

    let mut verifier_transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    let verifier_params = params.verifier_params();
    let strategy = SingleStrategy::new(&verifier_params);
    let result = verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<Bn256>, _, _, _>(
        &verifier_params,
        &vk,
        strategy,
        &[vec![]],
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "tampered proof should fail verification");
}

// A parallel MulCircuit + eltwise_mul_real_kzg_roundtrip test follows the exact same
// shape as AddCircuit/eltwise_add_real_kzg_roundtrip above, substituting
// EltwiseMulChip/EltwiseMulConfig and I18::from_f64 values whose product doesn't
// overflow (e.g. 2.0 * 3.0). Write it by mirroring the Add version — do not skip it,
// since Mul's quotient/remainder gate is the highest-risk piece in this sub-project.
```

The final comment block is a deliberate instruction, not a placeholder: mirror
`AddCircuit`/`eltwise_add_real_kzg_roundtrip`/`eltwise_add_tampered_proof_fails_verification`
into `MulCircuit`/`eltwise_mul_real_kzg_roundtrip`/`eltwise_mul_tampered_proof_fails_verification`,
swapping in `EltwiseMulChip`/`EltwiseMulConfig` and `EltwiseMulChip::configure`'s 5-argument
signature (`a, b, q, r, bits` columns instead of `a, b, c, bits`). This is mechanical
repetition of an already-fully-specified pattern, not an unresolved design question.

- [x] **Step 2: Run to verify it fails first (module doesn't exist / chips not public yet)**

Run: `cargo test -p zkie-core --test kzg_roundtrip`
Expected: FAIL initially if `chips`/`fixed_point`/`field_convert` modules or their contents
aren't `pub` yet — fix visibility (`pub mod`, `pub struct`, `pub fn`) as needed rather than
changing the test.

- [x] **Step 3: Run tests to verify they pass**

Run: `cargo test -p zkie-core --test kzg_roundtrip`
Expected: all 4 tests (`eltwise_add_real_kzg_roundtrip`, `eltwise_add_tampered_proof_fails_verification`,
`eltwise_mul_real_kzg_roundtrip`, `eltwise_mul_tampered_proof_fails_verification`) `ok`.

- [x] **Step 4: Run the full workspace test suite**

Run: `cargo test --workspace`
Expected: every test across `fixed_point`, `field_convert`, `range_check`, `eltwise`, `isa`,
`tensor`, and `kzg_roundtrip` passes.

- [x] **Step 5: Commit**

```bash
git add crates/zkie-core/tests/kzg_roundtrip.rs
git commit -m "test: add real KZG setup/prove/verify roundtrip for ELTWISE chips"
```

---

## Definition of Done

- [x] `cargo test --workspace` passes from `/Users/jimmyshi/code/zkie` (not `/tmp`) — 31 tests pass (27 unit + 4 KZG integration).
- [x] The KZG roundtrip test proves and verifies real (non-mocked) proofs for both
  `EltwiseAddChip` and `EltwiseMulChip`, plus tampered-proof negative cases.
- [x] `.spike-test/` (gitignored) deleted — it served its purpose of de-risking the halo2 API
  before writing this plan.
- [x] `cargo fmt --all` and `cargo clippy --workspace --all-targets` are clean (zero warnings).

## Status: Complete (2026-07-26)

All 7 tasks executed inline in this session (no user available to review between tasks —
they stepped away and asked for autonomous execution). Notable deviations from the plan's
exact code, discovered via real `cargo test` runs rather than guessed in advance:

- The `RangeCheckChip` running-sum gate's rotation offsets were initially reversed relative
  to `assign`'s row layout (row `i` holds bit `i`, LSB-first) — first `cargo test` run caught
  this via a `MockProver` constraint-not-satisfied failure with the exact cell values,
  which made the fix (`Rotation(-((n_bits - 1 - i) as i32))` instead of `Rotation(-(i as
  i32))`) obvious from the printed cell layout.
- `MockProver::run(k, ...)`'s `k` needed to be larger than planned: `k=8` was enough for a
  single 64-bit range check, but the Add chip's 3 range checks (a, b, c) together need more
  usable rows than `2^8` provides once blinding-factor rows are subtracted; bumped to `k=10`
  for all Eltwise-level tests.
- `EltwiseMulConfig` uses a dedicated `slack` column (not a second `RangeCheckChip` reusing
  the `r` column) for the `SCALE_18 - 1 - r` double-decomposition bound — the plan flagged
  this as a likely fallback if column-sharing caused a conflict, and it was simpler to just
  do it that way from the start.
- The Task 5 Step 5 negative tests (`add_with_forged_sum_is_rejected`,
  `mul_with_forged_quotient_is_rejected`) turned out not to need the "design note" fallback
  the plan sketched — `EltwiseAddConfig`/`EltwiseMulConfig`'s fields are private but visible
  to the same crate's `#[cfg(test)] mod tests`, so the tests access `config.add.s_add`,
  `config.add.a`, etc. directly without any exposed test-only accessors.

Everything else matches the plan as written and compiled/passed on the first or second
`cargo test` run per task.

## Addendum (2026-07-26): critical soundness fix discovered during sub-project 2

While building sub-project 2's `DivChip`, a subagent discovered that `RangeCheckChip::assign`
witnesses its own copy of the value being range-checked in a fresh region — and none of
`EltwiseAddChip`, `EltwiseMulChip`, `ReduceSumChip`, `ReduceMeanChip`, or `DotProductChip`
ever tied that witnessed cell back to the cell used in their own arithmetic gates via
`region.constrain_equal`. Confirmed exploitable with a direct `MockProver` probe: a proof
whose range-check regions held a disconnected, all-zero decoy witness verified successfully
alongside an unrelated, never-actually-checked value in the main gate.

All five affected chips were fixed (commits `d25b2f8`, `34c2533`, `3c14c43`): the column
holding a value that needs both to participate in a gate and to be range-checked now carries
its signed-shifted representation directly in the main gate's region (gate polynomials
adjusted to match), and the assigned cell is explicitly `region.constrain_equal`'d to
`RangeCheckChip`'s returned cell. Each chip gained a regression test that calls
`constrain_equal` with a deliberately mismatched decoy value and confirms the permutation
argument rejects it — proving the link is real, not just syntactically present (a naive
"assign a disconnected decoy and never call constrain_equal" test is not a meaningful probe,
since it doesn't exercise the mechanism under test at all).

`LookupChip` and `DivChip` were unaffected/already-fixed respectively: `LookupChip` doesn't
use `RangeCheckChip` (the lookup argument itself is the check, evaluated on the same cells
that are witnessed, so there's no cross-region disconnect possible), and `DivChip`'s own
implementing agent found and fixed this independently before the pattern was recognized as
crate-wide.

**Lesson for future chip work**: any time a value must both feed a polynomial gate and be
range-checked via `RangeCheckChip` (or any future composed sub-chip that witnesses its own
copy of a value), the two cells must be tied together with `region.constrain_equal` —
this is not optional plumbing, it is the difference between a real range check and a
decorative one.
