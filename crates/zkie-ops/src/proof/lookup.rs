//! Multiplicity commitment and OOD binding for LKP, with error accounting.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::CanonicalWriter;
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, STATEMENT};
use zkie_engine::pcs::request::fold::{prove_fold, verify_fold, FoldClaim, FoldReceipt, FoldShape};
use zkie_engine::primitives::random::OsRandomSource;
use zkie_core::common::random::RandomSource;

use crate::proof::binding::{
    commit_rows_with_openings, verify_opening, OracleId, OracleOpening, OracleRegistry,
};
use crate::proof::request::lkp::{multiplicity_rows, LookupTable};
use crate::proof::sumcheck::{accounting, SumcheckAccounting};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiplicityProof {
    pub oracle: OracleId,
    pub openings: Vec<OracleOpening>,
    pub point: Vec<Fp3>,
    pub answer: Fp3,
    pub query_digest: Digest,
    pub accounting: SumcheckAccounting,
}

pub fn padded_rows(rows: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let mut padded = rows.to_vec();
    let target = padded.len().max(1).next_power_of_two();
    padded.resize(target, vec![0]);
    padded
}

fn mle_eval(values: &[u64], point: &[Fp3]) -> Result<Fp3> {
    if values.len() != 1usize.checked_shl(point.len() as u32).unwrap_or(0) {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (index, value) in values.iter().enumerate() {
        let mut weight = Fp3::ONE;
        for (bit, coordinate) in point.iter().enumerate() {
            let one_minus = Fp3::ONE.sub(coordinate);
            weight = weight.mul(if (index >> bit) & 1 == 1 {
                coordinate
            } else {
                &one_minus
            });
        }
        sum = sum.add(&weight.mul(&Fp3::from_u64(*value)));
    }
    Ok(sum)
}

fn query_binding(registry: &OracleRegistry, queries: &[OracleOpening]) -> Result<Digest> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(queries.len() as u64);
    for query in queries {
        let root = registry.get(query.oracle).ok_or(Error::ProverStateMissing)?;
        writer.put_u64(query.oracle.0);
        writer.put_u64(query.index);
        writer.put_digest(&root.root);
        writer.put_u64(root.count);
    }
    hash_labeled(tag_word(STATEMENT, 11), 0, None, &writer.into_bytes())
}

fn point_from_chain(
    chain: &mut Chain,
    root: &Digest,
    count: u64,
    coordinates: u32,
    query_digest: &Digest,
) -> Result<Vec<Fp3>> {
    let mut message = chain.message();
    for word in root.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().map_err(|_| Error::InvalidEncoding)?,
        ));
    }
    message.push_u64(count);
    message.push_u64(coordinates as u64);
    for word in query_digest.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().map_err(|_| Error::InvalidEncoding)?,
        ));
    }
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut point = Vec::with_capacity(coordinates as usize);
    for _ in 0..coordinates {
        point.push(chain.draw_e()?);
    }
    Ok(point)
}

pub fn commit_multiplicities(
    registry: &mut OracleRegistry,
    label: &Digest,
    table: &LookupTable,
    queries: &[OracleOpening],
    padding_keys: &[Vec<u64>],
) -> Result<(OracleId, Vec<OracleOpening>)> {
    let rows = multiplicity_rows(table, queries, padding_keys)?;
    let padded = padded_rows(&rows);
    commit_rows_with_openings(registry, label, &padded)
}

pub fn prove_multiplicity_ood(
    chain: &mut Chain,
    table: &LookupTable,
    queries: &[OracleOpening],
    padding_keys: &[Vec<u64>],
    registry: &OracleRegistry,
    oracle: OracleId,
    openings: &[OracleOpening],
) -> Result<MultiplicityProof> {
    let rows = multiplicity_rows(table, queries, padding_keys)?;
    let padded = padded_rows(&rows);
    if openings.len() != padded.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut words = Vec::with_capacity(padded.len());
    for (index, opening) in openings.iter().enumerate() {
        if opening.oracle != oracle || opening.index != index as u64 {
            return Err(Error::StatementMismatch);
        }
        let root = registry.get(oracle).ok_or(Error::ProverStateMissing)?;
        verify_opening(registry, opening, root.count)?;
        let encoded = padded[index].clone();
        if opening.values != encoded {
            return Err(Error::InvalidProof);
        }
        words.push(encoded.last().copied().unwrap_or(0));
    }
    let root = registry.get(oracle).ok_or(Error::ProverStateMissing)?;
    let coordinates = padded.len().max(1).trailing_zeros();
    let query_digest = query_binding(registry, queries)?;
    let point = point_from_chain(
        chain,
        &root.root,
        padded.len() as u64,
        coordinates,
        &query_digest,
    )?;
    let answer = mle_eval(&words, &point)?;
    Ok(MultiplicityProof {
        oracle,
        openings: openings.to_vec(),
        point,
        answer,
        query_digest,
        accounting: accounting(coordinates),
    })
}

pub fn verify_multiplicity_ood(
    chain: &mut Chain,
    table: &LookupTable,
    queries: &[OracleOpening],
    padding_keys: &[Vec<u64>],
    registry: &OracleRegistry,
    proof: &MultiplicityProof,
) -> Result<()> {
    let rows = multiplicity_rows(table, queries, padding_keys)?;
    let padded = padded_rows(&rows);
    if proof.openings.len() != padded.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut words = Vec::with_capacity(padded.len());
    for (index, opening) in proof.openings.iter().enumerate() {
        if opening.oracle != proof.oracle || opening.index != index as u64 {
            return Err(Error::InvalidProof);
        }
        let root = registry.get(proof.oracle).ok_or(Error::ProverStateMissing)?;
        verify_opening(registry, opening, root.count)?;
        if opening.values != padded[index] {
            return Err(Error::InvalidProof);
        }
        words.push(opening.values.last().copied().unwrap_or(0));
    }
    let root = registry.get(proof.oracle).ok_or(Error::ProverStateMissing)?;
    let coordinates = padded.len().max(1).trailing_zeros();
    let query_digest = query_binding(registry, queries)?;
    if query_digest != proof.query_digest {
        return Err(Error::InvalidProof);
    }
    let point = point_from_chain(
        chain,
        &root.root,
        padded.len() as u64,
        coordinates,
        &query_digest,
    )?;
    if point != proof.point {
        return Err(Error::InvalidProof);
    }
    if mle_eval(&words, &point)? != proof.answer {
        return Err(Error::InvalidProof);
    }
    if proof.accounting.rounds != coordinates || proof.accounting.field_bits != 3 * 64 {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Multiplicity planes and mask.
//
// Every multiplicity count is decomposed into bit planes after the OOD point
// has been derived. Each plane is committed as its own oracle; the mask oracle
// marks real rows versus padded rows. The verifier checks each plane bit is
// boolean, recombines the count, and checks the mask relation.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneProof {
    pub planes: Vec<OracleId>,
    pub plane_openings: Vec<Vec<OracleOpening>>,
    pub mask: OracleId,
    pub mask_openings: Vec<OracleOpening>,
    pub visible_rows: u64,
}

pub fn commit_count_planes(
    registry: &mut OracleRegistry,
    label: &Digest,
    counts: &[u64],
    visible_rows: u64,
) -> Result<PlaneProof> {
    if visible_rows as usize > counts.len() || counts.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let padded = padded_rows(
        &counts
            .iter()
            .map(|count| vec![*count])
            .collect::<Vec<_>>(),
    );
    let mut planes = Vec::new();
    let mut plane_openings = Vec::new();
    for bit in 0..64u32 {
        let rows: Vec<Vec<u64>> = padded
            .iter()
            .map(|row| vec![(row[0] >> bit) & 1])
            .collect();
        let (oracle, openings) = commit_rows_with_openings(registry, label, &rows)?;
        planes.push(oracle);
        plane_openings.push(openings);
    }
    let mask_rows: Vec<Vec<u64>> = (0..padded.len())
        .map(|index| vec![u64::from((index as u64) < visible_rows)])
        .collect();
    let (mask, mask_openings) = commit_rows_with_openings(registry, label, &mask_rows)?;
    Ok(PlaneProof {
        planes,
        plane_openings,
        mask,
        mask_openings,
        visible_rows,
    })
}

/// Verify the plane decomposition and mask of already-opened counts.
/// `counts` are the raw multiplicity openings (single-word rows).
pub fn verify_count_planes(
    registry: &OracleRegistry,
    counts: &[OracleOpening],
    proof: &PlaneProof,
) -> Result<()> {
    let padded_len = padded_rows(
        &counts
            .iter()
            .map(|opening| opening.values.clone())
            .collect::<Vec<_>>(),
    )
    .len();
    if counts.is_empty()
        || proof.plain_len()? != padded_len
        || proof.visible_rows as usize > counts.len()
        || proof.planes.len() != 64
    {
        return Err(Error::ShapeMismatch);
    }
    let mask_root = registry.get(proof.mask).ok_or(Error::ProverStateMissing)?;
    if proof.mask_openings.len() != padded_len || mask_root.count != padded_len as u64 {
        return Err(Error::ShapeMismatch);
    }
    for (index, opening) in proof.mask_openings.iter().enumerate() {
        if opening.oracle != proof.mask || opening.index != index as u64 {
            return Err(Error::InvalidProof);
        }
        verify_opening(registry, opening, mask_root.count)?;
        let expected = u64::from((index as u64) < proof.visible_rows);
        if opening.values != vec![expected] {
            return Err(Error::InvalidProof);
        }
    }
    let mut reconstructed = vec![0u64; padded_len];
    for (bit, (oracle, openings)) in proof
        .planes
        .iter()
        .zip(&proof.plane_openings)
        .enumerate()
    {
        let root = registry.get(*oracle).ok_or(Error::ProverStateMissing)?;
        if openings.len() != padded_len || root.count != padded_len as u64 {
            return Err(Error::ShapeMismatch);
        }
        for (index, opening) in openings.iter().enumerate() {
            if opening.oracle != *oracle || opening.index != index as u64 {
                return Err(Error::InvalidProof);
            }
            verify_opening(registry, opening, root.count)?;
            let bit_value = opening.values.first().copied().ok_or(Error::InvalidProof)?;
            if bit_value > 1 {
                return Err(Error::InvalidProof);
            }
            reconstructed[index] |= bit_value << bit;
        }
    }
    for (index, opening) in counts.iter().enumerate() {
        let count = opening.values.first().copied().ok_or(Error::InvalidProof)?;
        if reconstructed[index] != count {
            return Err(Error::InvalidProof);
        }
        if (index as u64) >= proof.visible_rows && count != 0 {
            return Err(Error::InvalidProof);
        }
    }
    Ok(())
}

impl PlaneProof {
    fn plain_len(&self) -> Result<usize> {
        Ok(self.plane_openings.first().map(Vec::len).unwrap_or(0))
    }
}

// ---------------------------------------------------------------------------
// Combined PSC / multiplicity / plane entry.
//
// The OOD point is drawn first (after the multiplicity commitment), then the
// count planes and mask are committed. One entry verifies the multiplicity
// answer, the boolean plane decomposition and the mask relation.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupPlanesProof {
    pub multiplicity: MultiplicityProof,
    pub planes: PlaneProof,
    pub fold: FoldReceipt,
    pub fold_shape: FoldShape,
    pub fold_target: Fp3,
}

pub fn prove_lookup_with_planes(
    chain: &mut Chain,
    table: &LookupTable,
    queries: &[OracleOpening],
    padding_keys: &[Vec<u64>],
    registry: &mut OracleRegistry,
    oracle: OracleId,
    openings: &[OracleOpening],
    plane_label: &Digest,
) -> Result<LookupPlanesProof> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Lookup);
    let rows = multiplicity_rows(table, queries, padding_keys)?;
    let padded = padded_rows(&rows);
    let counts: Vec<u64> = padded
        .iter()
        .map(|row| row.last().copied().unwrap_or(0))
        .collect();
    let (fold, fold_shape, fold_target) = prove_psc_fold(chain, &counts)?;
    let multiplicity =
        prove_multiplicity_ood(chain, table, queries, padding_keys, registry, oracle, openings)?;
    let planes = commit_count_planes(registry, plane_label, &counts, rows.len() as u64)?;
    Ok(LookupPlanesProof {
        multiplicity,
        planes,
        fold,
        fold_shape,
        fold_target,
    })
}

pub fn verify_lookup_with_planes(
    chain: &mut Chain,
    table: &LookupTable,
    queries: &[OracleOpening],
    padding_keys: &[Vec<u64>],
    registry: &OracleRegistry,
    proof: &LookupPlanesProof,
) -> Result<()> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Lookup);
    let rows = multiplicity_rows(table, queries, padding_keys)?;
    let padded = padded_rows(&rows);
    let counts_for_fold: Vec<Fp3> = padded
        .iter()
        .map(|row| Fp3::from_u64(row.last().copied().unwrap_or(0)))
        .collect();
    let expected_target = counts_for_fold
        .iter()
        .fold(Fp3::ZERO, |acc, value| acc.add(value));
    if expected_target != proof.fold_target || proof.fold.shape != proof.fold_shape {
        return Err(Error::InvalidProof);
    }
    verify_fold(&proof.fold_shape, proof.fold_target, &proof.fold, chain)?;
    verify_multiplicity_ood(chain, table, queries, padding_keys, registry, &proof.multiplicity)?;
    let padded = padded_rows(&rows);
    let counts: Vec<OracleOpening> = proof
        .multiplicity
        .openings
        .iter()
        .zip(&padded)
        .map(|(opening, row)| OracleOpening {
            oracle: opening.oracle,
            index: opening.index,
            values: vec![row.last().copied().unwrap_or(0)],
            path: opening.path.clone(),
        })
        .collect();
    verify_count_planes(registry, &counts, &proof.planes)
}


fn prove_psc_fold(chain: &mut Chain, counts: &[u64]) -> Result<(FoldReceipt, FoldShape, Fp3)> {
    if counts.is_empty() || !counts.len().is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    let k = counts.len().trailing_zeros();
    let len_zk = (2 * k as usize + 1).next_power_of_two();
    let shape = FoldShape::new(k, 0, len_zk)?;
    let mut source = OsRandomSource::new()?;
    let mut words = vec![0u8; (2 * shape.k as usize + 1) * 8];
    source.fill(b"psc-fold", 0, 0, &mut words)?;
    let coins: Vec<Fp3> = words
        .chunks_exact(8)
        .map(|chunk| Fp3::from_u64(u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8]))))
        .collect();
    let main: Vec<Fp3> = counts.iter().map(|count| Fp3::from_u64(*count)).collect();
    let target = main.iter().fold(Fp3::ZERO, |acc, value| acc.add(value));
    let claim = FoldClaim {
        main_rows: counts.len(),
        main_cols: 1,
        main,
        coefficients_main: vec![Fp3::ONE; counts.len()],
        mask: vec![Fp3::ZERO; len_zk],
        coefficients_mask: vec![Fp3::ZERO; len_zk],
        kept: Fp3::ZERO,
        target,
    };
    let receipt = prove_fold(&shape, &claim, &coins, chain)?;
    Ok((receipt, shape, target))
}
