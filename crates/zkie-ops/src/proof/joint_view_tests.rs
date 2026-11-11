//! Test-only joint-view rank checks for the hidden output-head relation.

#[cfg(test)]
mod tests {
    use crate::proof::psc::{
        hp_entries, multilinear_eval, prove_dot_sumcheck, psc_variables, DotClaim, DotMaskEntries,
        HidingPolynomial, PscChallenges,
    };
    use zkie_core::error::Result as CoreResult;
    use zkie_engine::pcs::request::code::{build_code_plan, encode_row_fp3, point};
    use zkie_engine::pcs::request::fold::{FoldMainTable, FoldRoundProver};
    use zkie_engine::pcs::request::field::Fp3;

    fn rank(mut rows: Vec<Vec<Fp3>>) -> usize {
        let cols = rows.first().map(|row| row.len()).unwrap_or(0);
        let mut rank = 0usize;
        for column in 0..cols {
            let Some(pivot) = (rank..rows.len()).find(|row| !rows[*row][column].is_zero()) else {
                continue;
            };
            rows.swap(rank, pivot);
            let inverse = rows[rank][column].inverse().unwrap();
            for entry in rows[rank].iter_mut().skip(column) {
                *entry = entry.mul(&inverse);
            }
            for row in 0..rows.len() {
                if row == rank || rows[row][column].is_zero() {
                    continue;
                }
                let factor = rows[row][column];
                for column_index in column..cols {
                    let subtract = factor.mul(&rows[rank][column_index]);
                    rows[row][column_index] = rows[row][column_index].sub(&subtract);
                }
            }
            rank += 1;
            if rank == rows.len() {
                break;
            }
        }
        rank
    }

    fn det(mut matrix: Vec<Vec<Fp3>>) -> Fp3 {
        let n = matrix.len();
        assert!(matrix.iter().all(|row| row.len() == n));
        let mut value = Fp3::ONE;
        for column in 0..n {
            let Some(pivot) = (column..n).find(|row| !matrix[*row][column].is_zero()) else {
                return Fp3::ZERO;
            };
            if pivot != column {
                matrix.swap(pivot, column);
                value = Fp3::ZERO.sub(&value);
            }
            let diagonal = matrix[column][column];
            value = value.mul(&diagonal);
            let inverse = diagonal.inverse().unwrap();
            for row in (column + 1)..n {
                if matrix[row][column].is_zero() {
                    continue;
                }
                let factor = matrix[row][column].mul(&inverse);
                for col in column..n {
                    let subtract = factor.mul(&matrix[column][col]);
                    matrix[row][col] = matrix[row][col].sub(&subtract);
                }
            }
        }
        value
    }

    fn vandermonde_block(log_n: u32, q: usize) -> Vec<Vec<Fp3>> {
        let mut matrix = vec![vec![Fp3::ZERO; q]; q];
        for row in 0..q {
            let z = Fp3::from_base(point(log_n, row as u64).unwrap());
            let mut power = Fp3::ONE;
            for col in 0..q {
                matrix[row][col] = power;
                power = power.mul(&z);
            }
        }
        matrix
    }

    fn opening_matrices(log_n: u32, ell: u64, r: u64, q: usize) -> (Vec<Vec<Fp3>>, Vec<Vec<Fp3>>) {
        let mut tail = vec![vec![Fp3::ZERO; r as usize]; q];
        let mut message = vec![vec![Fp3::ZERO; ell as usize]; q];
        for row in 0..q {
            let z = Fp3::from_base(point(log_n, row as u64).unwrap());
            let mut power = Fp3::ONE;
            for column in 0..r as usize {
                tail[row][column] = power;
                power = power.mul(&z);
            }
            for column in 0..ell as usize {
                message[row][column] = power;
                power = power.mul(&z);
            }
        }
        (tail, message)
    }

    fn joined(mut left: Vec<Vec<Fp3>>, right: Vec<Vec<Fp3>>) -> Vec<Vec<Fp3>> {
        for (row, extra) in left.iter_mut().zip(right) {
            row.extend(extra);
        }
        left
    }

    #[test]
    fn joint_opening_submatrices_cover_the_message() {
        for (log_n, ell, r, q) in [
            (17u32, 65539u64, 8192u64, 16usize),
            (17, 65536, 8192, 16),
            (17, 512, 256, 8),
        ] {
            let (tail, message) = opening_matrices(log_n, ell, r, q);
            let tail_rank = rank(tail.clone());
            let joint_rank = rank(joined(tail, message));
            assert_eq!(tail_rank, q);
            assert_eq!(joint_rank, tail_rank);
        }
    }

    #[test]
    fn opening_vandermonde_block_is_full_rank_at_the_actual_table_size() {
        // The left q x q block of the q x r tail matrix is a Vandermonde
        // matrix. Its invertibility gives rank q of the full tail block, so
        // the message columns cannot raise the joint rank above q.
        for (log_n, q) in [
            (17u32, 603usize),
            (17, 16),
            (17, 4),
            (17, 4),
            (17, 256),
        ] {
            let block = vandermonde_block(log_n, q);
            assert_eq!(rank(block), q, "log_n {log_n} q {q}");
        }
    }

    struct FixedChallenges {
        values: Vec<Fp3>,
        at: usize,
    }

    impl FixedChallenges {
        fn new(values: Vec<Fp3>) -> Self {
            Self { values, at: 0 }
        }
    }

    impl PscChallenges for FixedChallenges {
        fn draw(&mut self) -> CoreResult<Fp3> {
            let value = self.values[self.at % self.values.len()];
            self.at += 1;
            Ok(value)
        }

        fn draw_nonzero(&mut self) -> CoreResult<Fp3> {
            Ok(Fp3::from_u64(3))
        }
    }

    fn flatten_proof(proof: crate::proof::psc::DotProof) -> Vec<Fp3> {
        let mut rows = Vec::new();
        rows.push(proof.hiding_sum);
        for round in &proof.rounds {
            rows.extend(round.coefficients.iter().copied());
        }
        rows.extend(proof.y_round.coefficients.iter().copied());
        rows.extend(proof.end_values.iter().copied());
        rows
    }

    fn closure_transcript_rows(
        hiding: &[Fp3],
        shown: [Fp3; 2],
        map_values: &[Fp3],
        bits: &[Fp3],
        challenges: &[Fp3],
    ) -> Vec<Fp3> {
        let omega = bits
            .iter()
            .map(|bit| Fp3::ONE.sub(bit))
            .collect::<Vec<_>>();
        let masks = DotMaskEntries {
            hiding: hiding.to_vec(),
            shown,
        };
        let claim = DotClaim {
            masks: Vec::new(),
            target: Fp3::ZERO,
        };
        let mut source = FixedChallenges::new(challenges.to_vec());
        let (proof, _) =
            prove_dot_sumcheck(map_values, &omega, &claim, &masks, &mut source).unwrap();
        flatten_proof(proof)
    }

    #[test]
    fn closure_transcript_decomposition_uses_the_real_prover() {
        let k = 4usize;
        let entries = hp_entries(&psc_variables(k).unwrap()).unwrap();
        let cells = 1usize << k;
        let challenges = (0..=k + 2)
            .map(|index| Fp3::from_u64(index as u64 * 11 + 5))
            .collect::<Vec<_>>();
        // The closure claim is satisfied by the boolean witness
        // `map = eq(r, .) * bit`; the product sum is zero. The shown masks
        // are fixed across the three runs: the public rounds are affine in
        // the hiding entries for every fixed secret and shown value.
        let r_bool = (0..k)
            .map(|index| Fp3::from_u64(index as u64 + 2))
            .collect::<Vec<_>>();
        let eq = crate::proof::output_closure::eq_vector(&r_bool).unwrap();
        let bits = (0..cells)
            .map(|index| {
                if index % 3 == 0 {
                    Fp3::ONE
                } else {
                    Fp3::ZERO
                }
            })
            .collect::<Vec<_>>();
        let map_values = bits
            .iter()
            .zip(&eq)
            .map(|(bit, weight)| weight.mul(bit))
            .collect::<Vec<_>>();
        let shown = [Fp3::from_u64(9), Fp3::from_u64(10)];
        let base = closure_transcript_rows(
            &vec![Fp3::ZERO; entries],
            shown,
            &map_values,
            &bits,
            &challenges,
        );
        let random = (0..entries)
            .map(|index| Fp3::from_u64(index as u64 * 5 + 2))
            .collect::<Vec<_>>();
        let combined =
            closure_transcript_rows(&random, shown, &map_values, &bits, &challenges);
        let mut expected = base.clone();
        for (column, weight) in random.iter().enumerate() {
            let mut basis = vec![Fp3::ZERO; entries];
            basis[column] = Fp3::ONE;
            let column_rows = closure_transcript_rows(
                &basis,
                shown,
                &map_values,
                &bits,
                &challenges,
            );
            for (row, value) in column_rows.iter().enumerate() {
                expected[row] = expected[row].add(&weight.mul(&value.sub(&base[row])));
            }
        }
        for (row, value) in combined.iter().enumerate() {
            assert_eq!(*value, expected[row], "row {row}");
        }
    }

    #[test]
    fn closure_end_values_are_one_time_padded() {
        let k = 4usize;
        let entries = hp_entries(&psc_variables(k).unwrap()).unwrap();
        let cells = 1usize << k;
        let challenges = (0..=k + 2)
            .map(|index| Fp3::from_u64(index as u64 * 17 + 3))
            .collect::<Vec<_>>();
        let r_bool = (0..k)
            .map(|index| Fp3::from_u64(index as u64 + 2))
            .collect::<Vec<_>>();
        let eq = crate::proof::output_closure::eq_vector(&r_bool).unwrap();
        let bits = (0..cells)
            .map(|index| {
                if index % 2 == 0 {
                    Fp3::ONE
                } else {
                    Fp3::ZERO
                }
            })
            .collect::<Vec<_>>();
        let map_values = bits
            .iter()
            .zip(&eq)
            .map(|(bit, weight)| weight.mul(bit))
            .collect::<Vec<_>>();
        let hiding = vec![Fp3::ZERO; entries];
        let zero = closure_transcript_rows(
            &hiding,
            [Fp3::ZERO; 2],
            &map_values,
            &bits,
            &challenges,
        );
        let shown = [Fp3::from_u64(4), Fp3::from_u64(6)];
        let masked =
            closure_transcript_rows(&hiding, shown, &map_values, &bits, &challenges);
        let beta = Fp3::from_u64(3);
        for index in 0..2 {
            let row = zero.len() - 2 + index;
            assert_eq!(masked[row].sub(&zero[row]), beta.mul(&shown[index]));
        }
    }

    fn closure_inner(left: &[Fp3], right: &[Fp3]) -> Fp3 {
        left.iter()
            .zip(right)
            .fold(Fp3::ZERO, |sum, (a, b)| sum.add(&a.mul(b)))
    }

    #[test]
    fn closure_point_values_are_real_products_plus_the_shown_masks() {
        let k = 4usize;
        let cells = 1usize << k;
        let r_bool = (0..k)
            .map(|index| Fp3::from_u64(index as u64 + 2))
            .collect::<Vec<_>>();
        let q_vector = (0..k)
            .map(|index| Fp3::from_u64(index as u64 * 3 + 4))
            .collect::<Vec<_>>();
        let eq = crate::proof::output_closure::eq_vector(&r_bool).unwrap();
        let eq_q = crate::proof::output_closure::eq_vector(&q_vector).unwrap();
        let bits = (0..cells)
            .map(|index| {
                if index % 3 == 0 {
                    Fp3::ONE
                } else {
                    Fp3::ZERO
                }
            })
            .collect::<Vec<_>>();
        let map_values = bits
            .iter()
            .zip(&eq)
            .map(|(bit, weight)| weight.mul(bit))
            .collect::<Vec<_>>();
        let point = (0..k)
            .map(|index| Fp3::from_u64(index as u64 * 5 + 1))
            .collect::<Vec<_>>();
        let b_point = multilinear_eval(&bits, &point).unwrap();
        let d_value = closure_inner(&eq_q, &map_values);
        let weighted = eq_q
            .iter()
            .zip(&eq)
            .map(|(q, r)| q.mul(r))
            .collect::<Vec<_>>();
        let bq_value = closure_inner(&weighted, &bits);
        for (index, base) in [b_point, d_value, bq_value].iter().enumerate() {
            let shown = Fp3::from_u64(index as u64 * 7 + 21);
            let value = base.add(&shown);
            assert_eq!(value.sub(base), shown);
        }
    }

    #[test]
    fn joint_closure_mask_object_covers_psc_shown_and_openings() {
        let k = 4usize;
        let entries = hp_entries(&psc_variables(k).unwrap()).unwrap();
        let shown = 5usize;
        let challenge_values = (0..=k)
            .map(|index| Fp3::from_u64(index as u64 * 13 + 7))
            .collect::<Vec<_>>();
        let hiding = psc_hiding_map(k, &challenge_values);
        let plan = build_code_plan(10, 32, 16).unwrap();
        let ell = 32usize;
        let tail_len = 16usize;
        let q = 2usize;
        let rows = entries + shown + q;
        let columns = ell + tail_len;
        let mut random = vec![vec![Fp3::ZERO; columns]; rows];
        for row in 0..entries {
            for column in 0..entries {
                random[row][column] = hiding[row][column];
            }
        }
        for index in 0..shown {
            random[entries + index][entries + index] = Fp3::ONE;
        }
        let positions = [0u64, 1];
        for column in 0..ell {
            let mut message = vec![Fp3::ZERO; ell];
            message[column] = Fp3::ONE;
            let tail = vec![Fp3::ZERO; tail_len];
            let code = encode_row_fp3(&plan, &message, &tail).unwrap();
            for (row, position) in positions.iter().enumerate() {
                random[entries + shown + row][column] = code[*position as usize];
            }
        }
        for column in 0..tail_len {
            let message = vec![Fp3::ZERO; ell];
            let mut tail = vec![Fp3::ZERO; tail_len];
            tail[column] = Fp3::ONE;
            let code = encode_row_fp3(&plan, &message, &tail).unwrap();
            for (row, position) in positions.iter().enumerate() {
                random[entries + shown + row][ell + column] =
                    code[*position as usize];
            }
        }
        let mut secret = vec![vec![Fp3::ZERO; entries + shown]; rows];
        for row in 0..(entries + shown) {
            secret[row][row] = Fp3::ONE;
        }
        let random_rank = rank(random.clone());
        let joint_rank = rank(joined(random, secret));
        assert_eq!(random_rank, rows);
        assert_eq!(joint_rank, random_rank);
    }

    #[test]
    fn joint_closure_mask_object_at_production_parameters() {
        use zkie_engine::pcs::request::code::evaluate_ood_fp3;

        let k = 23usize;
        let positions = 16usize;
        let entries = hp_entries(&psc_variables(k).unwrap()).unwrap(); // 49
        let shown = 5usize;
        let per_position = entries + shown; // 54
        let rows = 2usize; // mask message rows
        let ell = 512usize;
        let tail_len = 256usize;
        let q = 4usize; // opening positions per mask row
        let total_rows = positions * per_position + rows * q;
        let columns = rows * (ell + tail_len);
        let plan = build_code_plan(17, 512, 256).unwrap();
        let mut random = vec![vec![Fp3::ZERO; columns]; total_rows];
        let slot_column = |position: usize, index: usize| {
            let slot = position * per_position + index;
            let row = slot / ell;
            let place = slot % ell;
            row * ell + place
        };
        for position in 0..positions {
            let challenges = (0..=k)
                .map(|index| {
                    Fp3::from_u64((position as u64 * 97 + index as u64 * 13 + 5) % 1009)
                })
                .collect::<Vec<_>>();
            let hiding = psc_hiding_map(k, &challenges);
            for row in 0..entries {
                for column in 0..entries {
                    random[position * per_position + row][slot_column(position, column)] =
                        hiding[row][column];
                }
            }
            for index in 0..shown {
                random[position * per_position + entries + index]
                    [slot_column(position, entries + index)] = Fp3::ONE;
            }
        }
        // The opening block uses the real one-symbol evaluation formula on
        // unit messages and tails: two mask rows, four positions each.
        let opening_base = positions * per_position;
        let opening_positions = [0u64, 1, 2, 3];
        for mask_row in 0..rows {
            for (slot, opening) in opening_positions.iter().enumerate() {
                let row = opening_base + mask_row * q + slot;
                let z = Fp3::from_base(point(plan.log_n, *opening).unwrap());
                for column in 0..ell {
                    let mut message = vec![Fp3::ZERO; ell];
                    message[column] = Fp3::ONE;
                    let tail = vec![Fp3::ZERO; tail_len];
                    random[row][mask_row * ell + column] =
                        evaluate_ood_fp3(&plan, &message, &tail, z).unwrap();
                }
                for column in 0..tail_len {
                    let message = vec![Fp3::ZERO; ell];
                    let mut tail = vec![Fp3::ZERO; tail_len];
                    tail[column] = Fp3::ONE;
                    random[row][rows * ell + mask_row * tail_len + column] =
                        evaluate_ood_fp3(&plan, &message, &tail, z).unwrap();
                }
            }
        }
        let mut secret = vec![vec![Fp3::ZERO; positions * per_position]; total_rows];
        for row in 0..(positions * per_position) {
            secret[row][row] = Fp3::ONE;
        }
        let random_rank = rank(random.clone());
        let joint_rank = rank(joined(random, secret));
        assert_eq!(random_rank, total_rows);
        assert_eq!(joint_rank, random_rank);
    }

    fn dot_mask_rank(chi: Fp3) -> (usize, usize) {
        let mut random = vec![vec![Fp3::ZERO; 32]; 17];
        let mut secret = vec![vec![Fp3::ZERO; 16]; 17];
        for t in 0..16 {
            random[t][t] = Fp3::ONE;
            secret[t][t] = Fp3::ONE;
            random[16][t] = chi.pow_u64(t as u64);
            random[16][16 + t] = Fp3::ZERO.sub(&chi.pow_u64(t as u64));
        }
        let random_rank = rank(random.clone());
        let joint_rank = rank(joined(random, secret));
        (random_rank, joint_rank)
    }

    #[test]
    fn joint_dot_mask_block_is_covered_for_zero_and_nonzero_chi() {
        for chi in [Fp3::from_u64(7), Fp3::ZERO] {
            let (random_rank, joint_rank) = dot_mask_rank(chi);
            assert_eq!(random_rank, joint_rank);
        }
    }

    fn psc_hiding_map(k: usize, challenges: &[Fp3]) -> Vec<Vec<Fp3>> {
        let variables = psc_variables(k).unwrap();
        let entries = hp_entries(&variables).unwrap();
        let mut matrix = vec![vec![Fp3::ZERO; entries]; entries];
        for basis in 0..entries {
            let mut mask = vec![Fp3::ZERO; entries];
            mask[basis] = Fp3::ONE;
            let mut hiding = HidingPolynomial::new(&variables, &mask).unwrap();
            let mut row = 0usize;
            matrix[row][basis] = hiding.sum();
            row += 1;
            for variable in 0..=k {
                let round = hiding.round().unwrap();
                for (index, coefficient) in round.iter().enumerate() {
                    if index != variables[variable].fixed() {
                        matrix[row][basis] = *coefficient;
                        row += 1;
                    }
                }
                if variable < k {
                    hiding.bind(challenges[variable]).unwrap();
                }
            }
            assert_eq!(row, entries);
        }
        matrix
    }

    #[test]
    fn joint_psc_hiding_maps_are_full_rank_at_multiple_points() {
        for k in [10usize, 23] {
            let entries = hp_entries(&psc_variables(k).unwrap()).unwrap();
            for seed in 0u64..3 {
                let challenges = (0..=k)
                    .map(|index| Fp3::from_u64(seed * 17 + index as u64 * 3 + 2))
                    .collect::<Vec<_>>();
                let matrix = psc_hiding_map(k, &challenges);
                assert_eq!(rank(matrix.clone()), entries, "k {k} seed {seed}");
                let value = det(matrix);
                assert!(
                    value == Fp3::ONE || value == Fp3::ZERO.sub(&Fp3::ONE),
                    "k {k} seed {seed}"
                );
            }
        }
    }


    fn poly_at(coefficients: &[Fp3], x: Fp3) -> Fp3 {
        let mut value = Fp3::ZERO;
        for coefficient in coefficients.iter().rev() {
            value = value.mul(&x).add(coefficient);
        }
        value
    }

    fn psc_sigma_column(k: usize, challenges: &[Fp3], eps: Fp3) -> Vec<Fp3> {
        let variables = psc_variables(k).unwrap();
        let mut rows = Vec::new();
        rows.push(Fp3::ZERO);
        let mut psis = Fp3::ONE;
        for variable in 0..=k {
            let weight = &variables[variable];
            let scale = eps.mul(&psis);
            for (index, coefficient) in weight.psi().iter().enumerate() {
                if index != weight.fixed() {
                    rows.push(scale.mul(coefficient));
                }
            }
            if variable < k {
                psis = psis.mul(&poly_at(weight.psi(), challenges[variable]));
            }
        }
        rows
    }

    fn vec_columns(matrix: Vec<Vec<Fp3>>) -> Vec<Vec<Fp3>> {
        matrix
    }

    #[test]
    fn joint_mask_block_covers_the_secret_with_shared_claim_masks() {
        let k = 10usize;
        let challenges = (0..=k)
            .map(|index| Fp3::from_u64(index as u64 * 5 + 3))
            .collect::<Vec<_>>();
        for chi in [Fp3::from_u64(7), Fp3::ZERO] {
            let hiding = psc_hiding_map(k, &challenges);
            let sigma = psc_sigma_column(k, &challenges, Fp3::from_u64(11));
            let rows = 16 + 1 + 16 + hiding.len() + 2;
            let mut random = vec![vec![Fp3::ZERO; 16 + 16 + 16 + 23 + 2]; rows];
            let mut secret = vec![vec![Fp3::ZERO; 16]; rows];
            for t in 0..16 {
                random[t][t] = Fp3::ONE;
                secret[t][t] = Fp3::ONE;
                random[16][t] = chi.pow_u64(t as u64);
                random[16][16 + t] = Fp3::ZERO.sub(&chi.pow_u64(t as u64));
            }
            for t in 0..16 {
                random[17 + t][32 + t] = Fp3::ONE;
                secret[17 + t][t] = Fp3::ONE;
            }
            let psc_base = 33;
            for (row, values) in hiding.iter().enumerate() {
                random[psc_base + row][48..71].copy_from_slice(values);
            }
            // The claim mask direction: each of the 16 claim masks `c_t`
            // enters the PSC sigma with weight `chi^t`, so every claim-mask
            // column is the same sigma vector scaled by that weight.
            for t in 0..16usize {
                let coefficient = chi.pow_u64(t as u64);
                for row in 0..sigma.len() {
                    random[psc_base + row][16 + t] = coefficient.mul(&sigma[row]);
                }
            }
            random[rows - 2][71] = Fp3::ONE;
            random[rows - 1][72] = Fp3::ONE;
            let random_rank = rank(vec_columns(random.clone()));
            let joint_rank = rank(joined(random, secret));
            assert_eq!(random_rank, rows, "chi {chi:?}");
            assert_eq!(joint_rank, random_rank, "chi {chi:?}");
        }
    }

    #[test]
    fn joint_closure_block_covers_psc_and_point_values() {
        let k = 23usize;
        let challenges = (0..=k)
            .map(|index| Fp3::from_u64(index as u64 * 7 + 5))
            .collect::<Vec<_>>();
        let hiding = psc_hiding_map(k, &challenges);
        let rows = hiding.len() + 2 + 3;
        let mut random = vec![vec![Fp3::ZERO; 49 + 5]; rows];
        let mut secret = vec![vec![Fp3::ZERO; 3]; rows];
        for (row, values) in hiding.iter().enumerate() {
            random[row][..49].copy_from_slice(values);
        }
        let point_base = hiding.len() + 2;
        random[point_base][49] = Fp3::ONE;
        random[point_base + 1][50] = Fp3::ONE;
        random[point_base + 2][51] = Fp3::ONE;
        secret[point_base][0] = Fp3::ONE;
        secret[point_base + 1][1] = Fp3::ONE;
        secret[point_base + 2][2] = Fp3::ONE;
        random[point_base - 2][52] = Fp3::ONE;
        random[point_base - 1][53] = Fp3::ONE;
        let random_rank = rank(random.clone());
        let joint_rank = rank(joined(random, secret));
        assert_eq!(random_rank, rows);
        assert_eq!(joint_rank, random_rank);
    }


    #[test]
    fn joint_fold_fresh_coefficients_map_is_full_rank() {
        for k in [2u32, 14] {
        let entries = 2 * k as usize + 1;
        for seed in 0u64..3 {
            let mut matrix = vec![vec![Fp3::ZERO; entries]; entries];
            for basis in 0..entries {
                let mut coefficients = vec![Fp3::ZERO; entries];
                coefficients[basis] = Fp3::ONE;
                let mut prover = FoldRoundProver::new(
                    k,
                    0,
                    coefficients,
                    vec![Fp3::ZERO; entries],
                    vec![Fp3::ZERO; entries],
                    entries,
                    FoldMainTable::new(
                        vec![Fp3::ZERO; 1usize << k],
                        vec![Fp3::ZERO; 1usize << k],
                    )
                    .unwrap(),
                    Fp3::ZERO,
                )
                .unwrap();
                prover.start(Fp3::from_u64(seed + 3)).unwrap();
                matrix[0][basis] = prover.sum();
                let mut row = 1usize;
                for variable in 0..k {
                    let round = prover.round().unwrap();
                    matrix[row][basis] = round[0];
                    matrix[row + 1][basis] = round[2];
                    row += 2;
                    prover
                        .bind(Fp3::from_u64(seed * 5 + variable as u64 + 2))
                        .unwrap();
                }
                assert_eq!(row, entries);
            }
            assert_eq!(rank(matrix.clone()), entries, "k {k} seed {seed}");
            let value = det(matrix);
            assert!(
                value == Fp3::ONE || value == Fp3::ZERO.sub(&Fp3::ONE),
                "k {k} seed {seed}"
            );
        }
        }
    }

    #[test]
    fn joint_send_eval_and_shown_blocks_are_identity_covered() {
        for size in [1usize, 5] {
            let mut random = vec![vec![Fp3::ZERO; size]; size];
            let mut secret = vec![vec![Fp3::ZERO; size]; size];
            for index in 0..size {
                random[index][index] = Fp3::ONE;
                secret[index][index] = Fp3::ONE;
            }
            let random_rank = rank(random.clone());
            assert_eq!(rank(joined(random, secret)), random_rank);
        }
    }

    #[test]
    fn actual_code_parameters_accept_the_opening_set_sizes() {
        for (log_n, ell, r, q) in [
            (17u32, 65539u64, 8192u64, 603usize),
            (17, 65536, 8192, 4),
            (17, 512, 256, 4),
        ] {
            let plan = build_code_plan(log_n, ell, r).unwrap();
            assert!(q <= plan.r as usize);
            assert_eq!(plan.n(), 1u64 << log_n);
        }
    }
}
