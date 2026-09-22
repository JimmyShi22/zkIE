//! Non-native Goldilocks sum-check verifier in the BN254 scalar field.
//!
//! Verifies one round of the GKR sum-check (`t = 1`): given a round polynomial
//! `p(x) = c0 + c1*x + c2*x^2` over Goldilocks, a challenge `r`, a prior claim
//! `claimed`, and the two terminal evaluations `f`, `h`, it checks
//!
//!   * `p(0) + p(1) == claimed`  i.e. `2*c0 + c1 + c2 == claimed (mod p)`
//!   * `p(r) == f * h`           i.e. `c0 + c1*r + c2*r^2 == f*h (mod p)`
//!
//! Everything is computed non-natively: each Goldilocks value is witnessed as a
//! `u64` (64-bit range-checked) and every add/mul carries an explicit quotient or
//! overflow so the reduction by `p` is enforced inside the `Fr` circuit.

use crate::field_convert::Fr;
use crate::goldilocks::{add_mod, mul_mod, GoldilocksAddChip, GoldilocksMulChip};
use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner};
use halo2_proofs::plonk::{Circuit, Column, ConstraintSystem, ErrorFront, Instance};

#[derive(Clone, Debug)]
pub struct SumcheckConfig {
    add: crate::goldilocks::GoldilocksAddConfig,
    mul: crate::goldilocks::GoldilocksMulConfig,
    instance: Column<Instance>,
}

/// A single-round sum-check claim, all values as canonical `u64`s.
#[derive(Clone, Copy, Debug)]
pub struct SumcheckClaim {
    pub c0: u64,
    pub c1: u64,
    pub c2: u64,
    pub r: u64,
    pub claimed: u64,
    pub f: u64,
    pub h: u64,
}

/// Host-side evaluation of the round polynomial at `x`.
fn eval_poly(c0: u64, c1: u64, c2: u64, x: u64) -> u64 {
    let x2 = mul_mod(x, x).0;
    let t1 = mul_mod(c1, x).0;
    let t2 = mul_mod(c2, x2).0;
    let t3 = add_mod(c0, t1).0;
    add_mod(t3, t2).0
}

/// Host-side: is this a valid single-round sum-check transcript?
pub fn claim_is_valid(claim: &SumcheckClaim) -> bool {
    // p(0) + p(1) == claimed
    let s1 = add_mod(claim.c0, claim.c0).0;
    let s2 = add_mod(s1, claim.c1).0;
    let s3 = add_mod(s2, claim.c2).0;
    let lhs_ok = s3 == claim.claimed;
    // p(r) == f*h
    let pr = eval_poly(claim.c0, claim.c1, claim.c2, claim.r);
    let fh = mul_mod(claim.f, claim.h).0;
    lhs_ok && pr == fh
}

struct SumcheckCircuit {
    claim: SumcheckClaim,
}

impl Circuit<Fr> for SumcheckCircuit {
    type Config = SumcheckConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        SumcheckCircuit {
            claim: SumcheckClaim {
                c0: 0,
                c1: 0,
                c2: 0,
                r: 0,
                claimed: 0,
                f: 0,
                h: 0,
            },
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        SumcheckConfig {
            add: GoldilocksAddChip::configure(meta),
            mul: GoldilocksMulChip::configure(meta),
            instance,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let claim = self.claim;

        // p(0) + p(1) == claimed:  2c0 + c1 + c2 == claimed (mod p)
        let s1 = add_mod(claim.c0, claim.c0);
        let s2 = add_mod(s1.0, claim.c1);
        let s3 = add_mod(s2.0, claim.c2);
        GoldilocksAddChip::construct(config.add.clone()).assign(
            layouter.namespace(|| "s1 = 2c0"),
            claim.c0,
            claim.c0,
            s1.0,
            s1.1,
        )?;
        GoldilocksAddChip::construct(config.add.clone()).assign(
            layouter.namespace(|| "s2 = 2c0 + c1"),
            s1.0,
            claim.c1,
            s2.0,
            s2.1,
        )?;
        let s3_cell = GoldilocksAddChip::construct(config.add.clone()).assign(
            layouter.namespace(|| "s3 = 2c0 + c1 + c2"),
            s2.0,
            claim.c2,
            s3.0,
            s3.1,
        )?;

        // p(r) == f*h:  c0 + c1*r + c2*r^2 == f*h (mod p)
        let r2 = mul_mod(claim.r, claim.r);
        let u1 = mul_mod(claim.c1, claim.r);
        let u2 = mul_mod(claim.c2, r2.0);
        let u3 = add_mod(claim.c0, u1.0);
        let pr = add_mod(u3.0, u2.0);
        let fh = mul_mod(claim.f, claim.h);
        GoldilocksMulChip::construct(config.mul.clone()).assign(
            layouter.namespace(|| "r2 = r*r"),
            claim.r,
            claim.r,
            r2.0,
            r2.1,
        )?;
        GoldilocksMulChip::construct(config.mul.clone()).assign(
            layouter.namespace(|| "u1 = c1*r"),
            claim.c1,
            claim.r,
            u1.0,
            u1.1,
        )?;
        GoldilocksMulChip::construct(config.mul.clone()).assign(
            layouter.namespace(|| "u2 = c2*r2"),
            claim.c2,
            r2.0,
            u2.0,
            u2.1,
        )?;
        GoldilocksAddChip::construct(config.add.clone()).assign(
            layouter.namespace(|| "u3 = c0 + c1*r"),
            claim.c0,
            u1.0,
            u3.0,
            u3.1,
        )?;
        let pr_cell = GoldilocksAddChip::construct(config.add.clone()).assign(
            layouter.namespace(|| "pr = u3 + c2*r2"),
            u3.0,
            u2.0,
            pr.0,
            pr.1,
        )?;
        let fh_cell = GoldilocksMulChip::construct(config.mul).assign(
            layouter.namespace(|| "fh = f*h"),
            claim.f,
            claim.h,
            fh.0,
            fh.1,
        )?;

        // p(0) + p(1) == claimed, and p(r) == f*h.
        layouter.constrain_instance(s3_cell.cell(), config.instance, 0)?;
        layouter.assign_region(
            || "constrain pr == fh",
            |mut region| {
                region.constrain_equal(pr_cell.cell(), fh_cell.cell())?;
                Ok(())
            },
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goldilocks::GOLDILOCKS_P;
    use halo2_proofs::dev::MockProver;
    use rand_core::OsRng;

    fn rng_u64() -> u64 {
        use rand_core::RngCore;
        OsRng.next_u64() % GOLDILOCKS_P
    }

    fn valid_claim() -> SumcheckClaim {
        let c0 = rng_u64();
        let c1 = rng_u64();
        let c2 = rng_u64();
        let r = rng_u64();
        let f = rng_u64();
        let h = rng_u64();
        // claimed = p(0) + p(1) = 2c0 + c1 + c2
        let s1 = add_mod(c0, c0).0;
        let s2 = add_mod(s1, c1).0;
        let claimed = add_mod(s2, c2).0;
        SumcheckClaim {
            c0,
            c1,
            c2,
            r,
            claimed,
            f,
            h,
        }
    }

    #[test]
    fn valid_round_transcript_is_satisfied() {
        let claim = valid_claim();
        // f, h are random, so p(r) == f*h only holds with negligible probability.
        // Build a *valid* transcript by setting f*h = p(r).
        let pr = eval_poly(claim.c0, claim.c1, claim.c2, claim.r);
        // force f*h == pr by setting h = pr / f (choose f invertible-ish; simplest:
        // set f = 1, h = pr).
        let claim = SumcheckClaim { f: 1, h: pr, ..claim };
        assert!(claim_is_valid(&claim));
        let prover = MockProver::run(
            14,
            &SumcheckCircuit { claim },
            vec![vec![Fr::from(claim.claimed)]],
        )
        .unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn tampered_claimed_is_rejected() {
        let mut claim = valid_claim();
        let pr = eval_poly(claim.c0, claim.c1, claim.c2, claim.r);
        claim.f = 1;
        claim.h = pr;
        claim.claimed = add_mod(claim.claimed, 1).0; // corrupt the claim
        assert!(!claim_is_valid(&claim));
        let prover = MockProver::run(
            14,
            &SumcheckCircuit { claim },
            vec![vec![Fr::from(claim.claimed)]],
        )
        .unwrap();
        assert!(prover.verify().is_err());
    }
}
