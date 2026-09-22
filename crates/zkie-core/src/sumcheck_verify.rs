//! Non-native Goldilocks sum-check verifier in the BN254 scalar field.
//!
//! Verifies a GKR sum-check transcript over any number of rounds. Per round
//! `p_i(x) = c0 + c1*x + c2*x^2` with challenge `r`, the verifier chains two
//! checks: `p_i(0) + p_i(1) == prev` and `prev = p_i(r)`, then finally checks
//! `prev == f*h`. Every value is a canonical `u64` and every add/mul carries an
//! explicit overflow/quotient so the reduction by `p = 2^64 - 2^32 + 1` is
//! enforced non-natively inside the `Fr` circuit.

use crate::field_convert::Fr;
use crate::goldilocks::{add_mod, mul_mod, GoldilocksAddChip, GoldilocksMulChip};
use halo2_proofs::circuit::{AssignedCell, Layouter, SimpleFloorPlanner};
use halo2_proofs::plonk::{Circuit, Column, ConstraintSystem, ErrorFront, Instance};

#[derive(Clone, Debug)]
pub struct SumcheckConfig {
    add: crate::goldilocks::GoldilocksAddConfig,
    mul: crate::goldilocks::GoldilocksMulConfig,
    instance: Column<Instance>,
}

/// One sum-check round: the degree-2 polynomial and its challenge.
#[derive(Clone, Copy, Debug)]
pub struct RoundClaim {
    pub c0: u64,
    pub c1: u64,
    pub c2: u64,
    pub r: u64,
}

/// A full sum-check claim: all rounds, the initial claim, and the terminal
/// evaluations `f`, `h`.
#[derive(Clone, Debug)]
pub struct SumcheckClaim {
    pub rounds: Vec<RoundClaim>,
    pub claimed: u64,
    pub f: u64,
    pub h: u64,
}

/// Host-side `p(x) = c0 + c1*x + c2*x^2` over Goldilocks.
fn eval_poly(c0: u64, c1: u64, c2: u64, x: u64) -> u64 {
    let x2 = mul_mod(x, x).0;
    let t1 = mul_mod(c1, x).0;
    let t2 = mul_mod(c2, x2).0;
    let t3 = add_mod(c0, t1).0;
    add_mod(t3, t2).0
}

/// Host-side: is this a valid sum-check transcript?
pub fn claim_is_valid(claim: &SumcheckClaim) -> bool {
    let mut prev = claim.claimed;
    for round in &claim.rounds {
        let s1 = add_mod(round.c0, round.c0).0;
        let s2 = add_mod(s1, round.c1).0;
        let s3 = add_mod(s2, round.c2).0;
        if s3 != prev {
            return false;
        }
        prev = eval_poly(round.c0, round.c1, round.c2, round.r);
    }
    prev == mul_mod(claim.f, claim.h).0
}

/// The Halo2 circuit verifying a [`SumcheckClaim`] (exposed so integration tests
/// and the eventual real prover can instantiate it directly).
pub struct SumcheckCircuit {
    pub claim: SumcheckClaim,
}

impl Circuit<Fr> for SumcheckCircuit {
    type Config = SumcheckConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        SumcheckCircuit {
            claim: SumcheckClaim {
                rounds: Vec::new(),
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
        let add_chip = GoldilocksAddChip::construct(config.add.clone());
        let mul_chip = GoldilocksMulChip::construct(config.mul.clone());

        let mut s3_cells: Vec<AssignedCell<Fr, Fr>> = Vec::with_capacity(self.claim.rounds.len());
        let mut prev_cells: Vec<AssignedCell<Fr, Fr>> = Vec::with_capacity(self.claim.rounds.len());

        for (i, round) in self.claim.rounds.iter().enumerate() {
            // p(0) + p(1):  2c0 + c1 + c2
            let s1 = add_mod(round.c0, round.c0);
            let s2 = add_mod(s1.0, round.c1);
            let s3 = add_mod(s2.0, round.c2);
            add_chip.assign(
                layouter.namespace(|| format!("s1 r{i}")),
                round.c0,
                round.c0,
                s1.0,
                s1.1,
            )?;
            add_chip.assign(
                layouter.namespace(|| format!("s2 r{i}")),
                s1.0,
                round.c1,
                s2.0,
                s2.1,
            )?;
            let s3_cell = add_chip.assign(
                layouter.namespace(|| format!("s3 r{i}")),
                s2.0,
                round.c2,
                s3.0,
                s3.1,
            )?;
            s3_cells.push(s3_cell);

            // p(r) = c0 + c1*r + c2*r^2
            let r2 = mul_mod(round.r, round.r);
            let u1 = mul_mod(round.c1, round.r);
            let u2 = mul_mod(round.c2, r2.0);
            let u3 = add_mod(round.c0, u1.0);
            let np = add_mod(u3.0, u2.0);
            mul_chip.assign(
                layouter.namespace(|| format!("r2 r{i}")),
                round.r,
                round.r,
                r2.0,
                r2.1,
            )?;
            mul_chip.assign(
                layouter.namespace(|| format!("u1 r{i}")),
                round.c1,
                round.r,
                u1.0,
                u1.1,
            )?;
            mul_chip.assign(
                layouter.namespace(|| format!("u2 r{i}")),
                round.c2,
                r2.0,
                u2.0,
                u2.1,
            )?;
            add_chip.assign(
                layouter.namespace(|| format!("u3 r{i}")),
                round.c0,
                u1.0,
                u3.0,
                u3.1,
            )?;
            let np_cell = add_chip.assign(
                layouter.namespace(|| format!("np r{i}")),
                u3.0,
                u2.0,
                np.0,
                np.1,
            )?;
            prev_cells.push(np_cell);
        }

        // Final: prev == f*h.
        let fh = mul_mod(self.claim.f, self.claim.h);
        let fh_cell = mul_chip.assign(
            layouter.namespace(|| "fh"),
            self.claim.f,
            self.claim.h,
            fh.0,
            fh.1,
        )?;

        // Bind the chain: round 0's check against the public claim, then each
        // round against the previous round's evaluation, then the final product.
        layouter.constrain_instance(s3_cells[0].cell(), config.instance, 0)?;
        for i in 1..s3_cells.len() {
            layouter.assign_region(
                || format!("bind round {i}"),
                |mut region| {
                    region.constrain_equal(s3_cells[i].cell(), prev_cells[i - 1].cell())?;
                    Ok(())
                },
            )?;
        }
        layouter.assign_region(
            || "bind final",
            |mut region| {
                region.constrain_equal(prev_cells.last().unwrap().cell(), fh_cell.cell())?;
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
    use rand_core::{OsRng, RngCore};

    fn rng_u64() -> u64 {
        OsRng.next_u64() % GOLDILOCKS_P
    }

    /// Build a valid transcript backward from random challenges and terminal
    /// evaluations so the checks hold by construction.
    fn valid_claim(rounds: usize) -> SumcheckClaim {
        let f = rng_u64();
        let h = rng_u64();
        let mut prev = mul_mod(f, h).0;
        let mut rs = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let c0 = rng_u64();
            let c1 = rng_u64();
            let r = rng_u64();
            // Solve c2 so p(r) = prev, i.e. c2 = (prev - c0 - c1*r) / r^2.
            let r2 = mul_mod(r, r).0;
            let c1r = mul_mod(c1, r).0;
            // numerator = prev - c0 - c1*r  (mod p)
            let num = add_mod(prev, mul_mod(GOLDILOCKS_P - c0 % GOLDILOCKS_P, 1).0).0; // prev - c0
            let num = add_mod(num, mul_mod(GOLDILOCKS_P - c1r % GOLDILOCKS_P, 1).0).0; // - c1r
            // c2 = num / r^2 (mod p). r^2 invertible unless r=0; avoid r=0.
            let r2_inv = inverse(r2);
            let c2 = mul_mod(num, r2_inv).0;
            let round = RoundClaim { c0, c1, c2, r };
            rs.push(round);
            prev = add_mod(c0, c0).0;
            prev = add_mod(prev, c1).0;
            prev = add_mod(prev, c2).0; // p(0) + p(1)
        }
        // `prev` is now the required initial claim.
        rs.reverse();
        SumcheckClaim {
            rounds: rs,
            claimed: prev,
            f,
            h,
        }
    }

    /// Modular inverse of a Goldilocks element via Fermat (`p-2`).
    fn inverse(x: u64) -> u64 {
        let p = GOLDILOCKS_P;
        let mut base = x;
        let mut e = p - 2;
        let mut acc = 1u64;
        while e > 0 {
            if e & 1 == 1 {
                acc = mul_mod(acc, base).0;
            }
            base = mul_mod(base, base).0;
            e >>= 1;
        }
        acc
    }

    #[test]
    fn valid_multi_round_transcript_is_satisfied() {
        let claim = valid_claim(3);
        assert!(claim_is_valid(&claim));
        let prover = MockProver::run(
            16,
            &SumcheckCircuit {
                claim: claim.clone(),
            },
            vec![vec![Fr::from(claim.claimed)]],
        )
        .unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn tampered_terminal_product_is_rejected() {
        let mut claim = valid_claim(3);
        claim.f = add_mod(claim.f, 1).0; // corrupt f -> final product no longer matches
        assert!(!claim_is_valid(&claim));
        let prover = MockProver::run(
            16,
            &SumcheckCircuit {
                claim: claim.clone(),
            },
            vec![vec![Fr::from(claim.claimed)]],
        )
        .unwrap();
        assert!(prover.verify().is_err());
    }
}
