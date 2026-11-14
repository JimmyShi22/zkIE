//! The batched stack of the request relation's panels.
//!
//! The layer relation reduces its panels together: one masked sumcheck over the
//! union of the panels' variable cubes, their offsets assigned largest panel
//! first, and one end value for each side. This module fixes the panel layout
//! and the verifier's affine coefficients at the stack's challenge point:
//! `A0` and `B0` are the shown end values, and the affine forms `Alin` and
//! `Blin` recombine the panels' operand values so that the verifier can tie
//! the end values to the reads.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;

use crate::proof::psc::{fold_first, product_round};
use crate::proof::psc_general::{
    prove_general, verify_general, GeneralChallenges, GeneralMaskEntries, GeneralOutput,
    GeneralProof, InnerSumcheck, MaskTerm, PanelEnd, Summand,
};
use crate::proof::request_weights::eq1;

/// One two-factor panel: `2^w` aligned points, `ni` rows of `ny` entries, a
/// row weight each, and the claim it belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackPanel {
    /// Index of the batched claim the panel belongs to.
    pub claim: usize,
    /// The panel's call factor.
    pub ec: Fp3,
    pub ni: u64,
    pub ny: u64,
    pub weight: Vec<Fp3>,
    /// The first weight claim of a weight panel, if the second side is weights.
    pub first_claim: Option<u32>,
    /// Assigned by `panel_layout`.
    pub w: u32,
    pub off: u64,
}

fn ceil_log2(value: u64) -> u32 {
    if value <= 1 {
        0
    } else {
        64 - (value - 1).leading_zeros()
    }
}

/// Assigns every panel its width and offset: largest panels first, each
/// aligned to its own size, over whole powers of two. Returns the stack's total
/// width in bits.
pub fn panel_layout(panels: &mut [StackPanel]) -> Result<u32> {
    if panels.is_empty() {
        return Err(Error::InvalidConfig);
    }
    for panel in panels.iter() {
        if !panel.ni.is_power_of_two()
            || !panel.ny.is_power_of_two()
            || panel.weight.len() as u64 != panel.ni
        {
            return Err(Error::InvalidConfig);
        }
    }
    let mut order: Vec<usize> = (0..panels.len()).collect();
    order.sort_by_key(|index| std::cmp::Reverse(ceil_log2(panels[*index].ni) + ceil_log2(panels[*index].ny)));
    let mut offset = 0u64;
    for index in order {
        let w = ceil_log2(panels[index].ni) + ceil_log2(panels[index].ny);
        panels[index].w = w;
        panels[index].off = offset;
        offset = offset
            .checked_add(1u64 << w)
            .ok_or(Error::ResourceLimit)?;
    }
    if offset > (1u64 << 20) {
        return Err(Error::ResourceLimit);
    }
    Ok(ceil_log2(offset))
}

/// The equality factors of a range of the challenge coordinates: entry `index`
/// pairs bit `j` of `index` with `rs[lo + j]`.
pub fn eq_low(rs: &[Fp3], lo: usize, hi: usize) -> Result<Vec<Fp3>> {
    if lo >= hi || hi > rs.len() || hi - lo >= 32 {
        return Err(Error::InvalidConfig);
    }
    let count = 1usize << (hi - lo);
    let mut table = vec![Fp3::ONE; count];
    for j in 0..(hi - lo) {
        for (index, value) in table.iter_mut().enumerate() {
            let chosen = if (index >> j) & 1 == 1 {
                rs[lo + j]
            } else {
                Fp3::ONE.sub(&rs[lo + j])
            };
            *value = value.mul(&chosen);
        }
    }
    Ok(table)
}

/// One equality factor of a range at an index.
pub fn eq_low_at(rs: &[Fp3], lo: usize, hi: usize, index: u64) -> Result<Fp3> {
    if lo >= hi || hi > rs.len() || lo >= 32 {
        return Err(Error::InvalidConfig);
    }
    let mut value = Fp3::ONE;
    for j in 0..(hi - lo) {
        value = value.mul(&eq1(rs[lo + j], if (index >> j) & 1 == 1 { Fp3::ONE } else { Fp3::ZERO }));
    }
    Ok(value)
}

/// One term of an affine form: the coefficient of the panel's operand value at
/// row `i` and entry `yv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffTerm {
    pub panel: usize,
    pub i: u64,
    pub yv: u64,
    pub coefficient: Fp3,
}

/// The verifier's recombination of the panels at the stack's point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackAffines {
    /// The first side's terms.
    pub alin: Vec<AffTerm>,
    /// The second side's terms for panels whose second side is a read.
    pub blin: Vec<AffTerm>,
    /// The second side's terms for weight panels: panel, row, coefficient of
    /// the shown weight value.
    pub shown: Vec<(usize, u64, Fp3)>,
}

/// The equality factors of a range that may be empty: an empty range has the
/// single empty product, one.
fn eq_range(point: &[Fp3], lo: usize, hi: usize) -> Result<Vec<Fp3>> {
    if lo == hi {
        Ok(vec![Fp3::ONE])
    } else {
        eq_low(point, lo, hi)
    }
}

/// Builds the stack's affine recombination. `rs` holds the stack's challenge
/// coordinates, `lam` its width in bits, `om` the batch coefficients of the
/// panels' claims.
pub fn stack_affines(
    panels: &[StackPanel],
    rs: &[Fp3],
    lam: u32,
    om: &[Fp3],
) -> Result<StackAffines> {
    if lam as usize != rs.len() || om.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut alin = Vec::new();
    let mut blin = Vec::new();
    let mut shown = Vec::new();
    for (index, panel) in panels.iter().enumerate() {
        if panel.w > lam || panel.off >> panel.w >= (1u64 << (lam - panel.w).min(63)) {
            return Err(Error::InvalidConfig);
        }
        let wy = ceil_log2(panel.ny) as usize;
        let w = panel.w as usize;
        if w < wy {
            return Err(Error::InvalidConfig);
        }
        let ey = eq_range(rs, 0, wy)?;
        let ei = eq_range(rs, wy, w)?;
        if ei.len() as u64 != panel.ni {
            return Err(Error::InvalidConfig);
        }
        let batch = *om.get(panel.claim).ok_or(Error::InvalidIndex)?;
        let sb = if panel.w == lam {
            Fp3::ONE
        } else {
            eq_low_at(rs, w, lam as usize, (panel.off >> panel.w) as u64)?
        };
        let co = sb.mul(&batch).mul(&panel.ec);
        for i in 0..panel.ni {
            for yv in 0..panel.ny {
                let factor = ei[i as usize].mul(&ey[yv as usize]);
                alin.push(AffTerm {
                    panel: index,
                    i,
                    yv,
                    coefficient: co.mul(&panel.weight[i as usize]).mul(&factor),
                });
            }
        }
        match panel.first_claim {
            None => {
                for i in 0..panel.ni {
                    for yv in 0..panel.ny {
                        blin.push(AffTerm {
                            panel: index,
                            i,
                            yv,
                            coefficient: sb.mul(&ei[i as usize]).mul(&ey[yv as usize]),
                        });
                    }
                }
            }
            Some(_) => {
                for i in 0..panel.ni {
                    shown.push((index, i, sb.mul(&ei[i as usize])));
                }
            }
        }
    }
    Ok(StackAffines { alin, blin, shown })
}

/// The stack as the reduction uses it: the panels, their layout and the two
/// sides' packed cube values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackReduction {
    pub lam: u32,
    pub panels: Vec<StackPanel>,
}

impl StackReduction {
    pub fn new(mut panels: Vec<StackPanel>) -> Result<Self> {
        let lam = panel_layout(&mut panels)?;
        Ok(Self { lam, panels })
    }

    /// The stack's claim: every panel's weighted operand product, with the
    /// call and batch factors.
    pub fn sum(
        &self,
        left: &[Vec<Fp3>],
        right: &[Vec<Fp3>],
        om: &[Fp3],
    ) -> Result<Fp3> {
        let (left_map, right_map) = self.side_maps(left, right, om)?;
        let mut sum = Fp3::ZERO;
        for (left, right) in left_map.iter().zip(&right_map) {
            sum = sum.add(&left.mul(right));
        }
        Ok(sum)
    }

    /// The two sides as cube vectors over the stack's `2^lam` points: a panel's
    /// cell `(i, yv)` sits at `off + (i << wy) + yv`, the first side carries the
    /// call factor, the batch coefficient and the row weight.
    pub fn side_maps(
        &self,
        left: &[Vec<Fp3>],
        right: &[Vec<Fp3>],
        om: &[Fp3],
    ) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
        if left.len() != self.panels.len() || right.len() != self.panels.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut left_map = vec![Fp3::ZERO; 1usize << self.lam];
        let mut right_map = vec![Fp3::ZERO; 1usize << self.lam];
        for (index, panel) in self.panels.iter().enumerate() {
            let cells = (panel.ni as usize)
                .checked_mul(panel.ny as usize)
                .ok_or(Error::ResourceLimit)?;
            if left[index].len() != cells || right[index].len() != cells {
                return Err(Error::ShapeMismatch);
            }
            let wy = ceil_log2(panel.ny) as usize;
            let batch = *om.get(panel.claim).ok_or(Error::InvalidIndex)?;
            let co = panel.ec.mul(&batch);
            for i in 0..panel.ni as usize {
                for yv in 0..panel.ny as usize {
                    let at = usize::try_from(panel.off).map_err(|_| Error::ResourceLimit)?
                        + (i << wy)
                        + yv;
                    let cell = i * panel.ny as usize + yv;
                    let slot = left_map.get_mut(at).ok_or(Error::InvalidIndex)?;
                    *slot = co.mul(&panel.weight[i]).mul(&left[index][cell]);
                    right_map[at] = right[index][cell];
                }
            }
        }
        Ok((left_map, right_map))
    }

    /// The multilinear extension of a cube vector at a point: bit `j` of the
    /// index pairs with `point[j]`.
    pub fn mle(&self, values: &[Fp3], point: &[Fp3]) -> Result<Fp3> {
        if point.len() != self.lam as usize || values.len() != 1usize << self.lam {
            return Err(Error::InvalidConfig);
        }
        let mut sum = Fp3::ZERO;
        for (index, value) in values.iter().enumerate() {
            let mut factor = Fp3::ONE;
            for (bit, coordinate) in point.iter().enumerate() {
                let chosen = if (index >> bit) & 1 == 1 {
                    *coordinate
                } else {
                    Fp3::ONE.sub(coordinate)
                };
                factor = factor.mul(&chosen);
            }
            sum = sum.add(&factor.mul(value));
        }
        Ok(sum)
    }

    /// The end values the prover sends: each side's packed map at the stack
    /// point.
    pub fn end_values(
        &self,
        left: &[Vec<Fp3>],
        right: &[Vec<Fp3>],
        om: &[Fp3],
        point: &[Fp3],
    ) -> Result<(Fp3, Fp3)> {
        let (left_map, right_map) = self.side_maps(left, right, om)?;
        Ok((self.mle(&left_map, point)?, self.mle(&right_map, point)?))
    }

    /// Checks the two row ties at the stack point: each sent end value equals
    /// its affine recombination of the panels' operand values.
    pub fn check_row_ties(
        &self,
        left: &[Vec<Fp3>],
        right: &[Vec<Fp3>],
        om: &[Fp3],
        point: &[Fp3],
        left_end: Fp3,
        right_end: Fp3,
    ) -> Result<()> {
        let affines = stack_affines(&self.panels, point, self.lam, om)?;
        let mut alin = Fp3::ZERO;
        for term in &affines.alin {
            let panel = &self.panels[term.panel];
            let cell = (term.i * panel.ny + term.yv) as usize;
            alin = alin.add(&term.coefficient.mul(&left[term.panel][cell]));
        }
        let mut blin = Fp3::ZERO;
        for term in &affines.blin {
            let panel = &self.panels[term.panel];
            let cell = (term.i * panel.ny + term.yv) as usize;
            blin = blin.add(&term.coefficient.mul(&right[term.panel][cell]));
        }
        for (panel_index, i, coefficient) in &affines.shown {
            let panel = &self.panels[*panel_index];
            let cell = (*i * panel.ny) as usize;
            blin = blin.add(&coefficient.mul(&right[*panel_index][cell]));
        }
        if left_end != alin || right_end != blin {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

/// One end value as it was sent: the shown value, its mask slot and the mask
/// factor the sumcheck applied to every end value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndValue {
    pub value: Fp3,
    pub slot: usize,
    pub scale: Fp3,
}

impl StackReduction {
    /// Checks the two end-value ties: an end value's hidden value (the shown
    /// value minus the factor times its mask entry) must equal the affine
    /// recombination of the panels' operand values, with the shown weights on
    /// the second side. The stack sumcheck's own leftover row claim is the
    /// third linear claim (`RowForm::from_output`).
    #[allow(clippy::too_many_arguments)]
    pub fn check_end_ties(
        &self,
        left: &[Vec<Fp3>],
        right: &[Vec<Fp3>],
        om: &[Fp3],
        point: &[Fp3],
        left_end: EndValue,
        right_end: EndValue,
        mask_value: &dyn Fn(usize) -> Result<Fp3>,
    ) -> Result<()> {
        let affines = stack_affines(&self.panels, point, self.lam, om)?;
        let mut alin = Fp3::ZERO;
        for term in &affines.alin {
            let panel = &self.panels[term.panel];
            let cell = (term.i * panel.ny + term.yv) as usize;
            alin = alin.add(&term.coefficient.mul(&left[term.panel][cell]));
        }
        let mut blin = Fp3::ZERO;
        for term in &affines.blin {
            let panel = &self.panels[term.panel];
            let cell = (term.i * panel.ny + term.yv) as usize;
            blin = blin.add(&term.coefficient.mul(&right[term.panel][cell]));
        }
        for (panel_index, i, coefficient) in &affines.shown {
            let panel = &self.panels[*panel_index];
            let cell = (*i * panel.ny) as usize;
            blin = blin.add(&coefficient.mul(&right[*panel_index][cell]));
        }
        let left_hidden = left_end
            .value
            .sub(&left_end.scale.mul(&mask_value(left_end.slot)?));
        let right_hidden = right_end
            .value
            .sub(&right_end.scale.mul(&mask_value(right_end.slot)?));
        if left_hidden != alin || right_hidden != blin {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

/// The hidden values of the stack's end group: the two sides' rest values and
/// the weight values. The order of `v` is the panels' order, rows within a
/// panel in increasing order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupValues {
    pub a0: Fp3,
    pub b0: Fp3,
    pub v: Vec<Fp3>,
}

impl GroupValues {
    /// Derives the hidden values from the sent ones: `hidden = sent - factor *
    /// mask entry`. `never_sent` is the number of the group's u values between
    /// A0 and B0; this consumer's group has none.
    pub fn from_sent(
        sent: &[Fp3],
        slots: &[usize],
        factor: Fp3,
        never_sent: usize,
        mask_value: &dyn Fn(usize) -> Result<Fp3>,
    ) -> Result<Self> {
        if sent.len() != slots.len() || sent.len() < 2 + never_sent {
            return Err(Error::ShapeMismatch);
        }
        let mut hidden = Vec::with_capacity(sent.len());
        for (value, slot) in sent.iter().zip(slots) {
            hidden.push(value.sub(&factor.mul(&mask_value(*slot)?)));
        }
        Ok(Self {
            a0: hidden[0],
            b0: hidden[1 + never_sent],
            v: hidden[2 + never_sent..].to_vec(),
        })
    }
}

impl StackReduction {
    /// Checks the group's source ties and returns the two sides the terminal
    /// relation multiplies: A0 against the reads' recombination on the first
    /// side, B0 against the second side's reads, and the weight values against
    /// the weight coefficients. The weight values' own claims are checked by
    /// the merge, not here.
    pub fn group_bases(
        &self,
        left: &[Vec<Fp3>],
        right: &[Vec<Fp3>],
        om: &[Fp3],
        point: &[Fp3],
        group: &GroupValues,
    ) -> Result<(Fp3, Fp3)> {
        let affines = stack_affines(&self.panels, point, self.lam, om)?;
        if affines.shown.len() != group.v.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut alin = Fp3::ZERO;
        for term in &affines.alin {
            let panel = &self.panels[term.panel];
            let cell = (term.i * panel.ny + term.yv) as usize;
            alin = alin.add(&term.coefficient.mul(&left[term.panel][cell]));
        }
        let mut blin = Fp3::ZERO;
        for term in &affines.blin {
            let panel = &self.panels[term.panel];
            let cell = (term.i * panel.ny + term.yv) as usize;
            blin = blin.add(&term.coefficient.mul(&right[term.panel][cell]));
        }
        if group.a0 != alin || group.b0 != blin {
            return Err(Error::StatementMismatch);
        }
        let mut bs = group.b0;
        for ((_, _, coefficient), value) in affines.shown.iter().zip(&group.v) {
            bs = bs.add(&coefficient.mul(value));
        }
        Ok((group.a0, bs))
    }
}

impl StackReduction {
    /// The weight panels' end geometry in the order their rows are sent.
    pub fn panel_ends(&self) -> Vec<PanelEnd> {
        self.panels
            .iter()
            .filter(|panel| panel.first_claim.is_some())
            .map(|panel| PanelEnd {
                off: panel.off,
                w: panel.w,
                ni: panel.ni,
                ny: panel.ny,
            })
            .collect()
    }

    /// The packed maps of the whole end group: the A side, the read-only B
    /// side and one map per weight-panel row, in panel and row order.
    pub fn group_maps(
        &self,
        left: &[Vec<Fp3>],
        right: &[Vec<Fp3>],
        om: &[Fp3],
    ) -> Result<(Vec<Fp3>, Vec<Fp3>, Vec<Vec<Fp3>>)> {
        if left.len() != self.panels.len() || right.len() != self.panels.len() {
            return Err(Error::ShapeMismatch);
        }
        let size = 1usize << self.lam;
        let mut a0 = vec![Fp3::ZERO; size];
        let mut b0 = vec![Fp3::ZERO; size];
        let mut v_maps = Vec::new();
        for (index, panel) in self.panels.iter().enumerate() {
            let cells = (panel.ni as usize)
                .checked_mul(panel.ny as usize)
                .ok_or(Error::ResourceLimit)?;
            if left[index].len() != cells || right[index].len() != cells {
                return Err(Error::ShapeMismatch);
            }
            let wy = ceil_log2(panel.ny) as usize;
            let batch = *om.get(panel.claim).ok_or(Error::InvalidIndex)?;
            let co = panel.ec.mul(&batch);
            for i in 0..panel.ni as usize {
                for yv in 0..panel.ny as usize {
                    let at = usize::try_from(panel.off).map_err(|_| Error::ResourceLimit)?
                        + (i << wy)
                        + yv;
                    let cell = i * panel.ny as usize + yv;
                    match panel.first_claim {
                        None => {
                            a0[at] = co.mul(&panel.weight[i]).mul(&left[index][cell]);
                            b0[at] = right[index][cell];
                        }
                        Some(_) => {
                            a0[at] = co.mul(&panel.weight[i]).mul(&left[index][cell]);
                        }
                    }
                }
            }
            if panel.first_claim.is_some() {
                for i in 0..panel.ni as usize {
                    let mut map = vec![Fp3::ZERO; size];
                    for yv in 0..panel.ny as usize {
                        let at = usize::try_from(panel.off).map_err(|_| Error::ResourceLimit)?
                            + (i << wy)
                            + yv;
                        map[at] = right[index][i * panel.ny as usize + yv];
                    }
                    v_maps.push(map);
                }
            }
        }
        Ok((a0, b0, v_maps))
    }
}

/// The group's inner sumcheck: the packed maps' product rounds and the whole
/// end group at the point. The anchor sends the unpacked weight row values, so
/// the inner records the challenges, recomputes each row's end coefficient at
/// the end and divides the packed row value by it.
pub struct GroupInner {
    a0: Vec<Fp3>,
    b0: Vec<Fp3>,
    v: Vec<Vec<Fp3>>,
    panels: Vec<PanelEnd>,
    lam: u32,
    challenges: Vec<Fp3>,
}

impl GroupInner {
    pub fn new(
        a0: Vec<Fp3>,
        b0: Vec<Fp3>,
        v: Vec<Vec<Fp3>>,
        panels: Vec<PanelEnd>,
        lam: u32,
    ) -> Result<Self> {
        if a0.len() < 2
            || !a0.len().is_power_of_two()
            || b0.len() != a0.len()
            || v.iter().any(|map| map.len() != a0.len())
            || panels.iter().map(|panel| panel.ni as usize).sum::<usize>() != v.len()
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            a0,
            b0,
            v,
            panels,
            lam,
            challenges: Vec::new(),
        })
    }

    fn right(&self) -> Vec<Fp3> {
        let mut right = self.b0.clone();
        for map in &self.v {
            for (slot, value) in right.iter_mut().zip(map) {
                *slot = slot.add(value);
            }
        }
        right
    }

    fn row_coefficient(&self, panel: &PanelEnd, row: usize) -> Result<Fp3> {
        let wy = ceil_log2(panel.ny) as usize;
        let w = panel.w as usize;
        if w < wy || w > self.lam as usize || self.challenges.len() != self.lam as usize {
            return Err(Error::InvalidConfig);
        }
        let sb = if panel.w == self.lam {
            Fp3::ONE
        } else {
            eq_low_at(
                &self.challenges,
                w,
                self.lam as usize,
                panel.off >> panel.w,
            )?
        };
        let ei = eq_range(&self.challenges, wy, w)?;
        ei.get(row).copied().ok_or(Error::ShapeMismatch).map(|value| sb.mul(&value))
    }
}

impl InnerSumcheck for GroupInner {
    fn round(&self) -> Result<Vec<Fp3>> {
        Ok(product_round(&self.a0, &self.right())?.to_vec())
    }

    fn bind(&mut self, challenge: Fp3) -> Result<()> {
        fold_first(&mut self.a0, challenge)?;
        fold_first(&mut self.b0, challenge)?;
        for map in &mut self.v {
            fold_first(map, challenge)?;
        }
        self.challenges.push(challenge);
        Ok(())
    }

    fn end_values(&self) -> Result<Vec<Fp3>> {
        if self.a0.len() != 1 || self.b0.len() != 1 || self.v.iter().any(|map| map.len() != 1) {
            return Err(Error::InvalidSessionState);
        }
        let a_star = self.a0[0];
        let b_star = self.right()[0];
        let mut values = Vec::with_capacity(2 + self.v.len());
        values.push(a_star);
        let mut rest = b_star;
        let mut row = 0usize;
        for panel in &self.panels {
            for i in 0..panel.ni as usize {
                let coefficient = self.row_coefficient(panel, i)?;
                if coefficient.is_zero() {
                    return Err(Error::InvalidConfig);
                }
                let packed = self.v[row][0];
                let unpacked = packed.mul(&coefficient.inverse()?);
                rest = rest.sub(&packed);
                values.push(unpacked);
                row += 1;
            }
        }
        values.insert(1, rest);
        Ok(values)
    }
}

/// Proves the stack's whole end group.
#[allow(clippy::too_many_arguments)]
pub fn prove_stack_group(
    reduction: &StackReduction,
    left: &[Vec<Fp3>],
    right: &[Vec<Fp3>],
    om: &[Fp3],
    claim_masks: &[MaskTerm],
    mask_value: &dyn Fn(usize) -> Result<Fp3>,
    target: Fp3,
    masks: &GeneralMaskEntries,
    challenges: &mut dyn GeneralChallenges,
) -> Result<(GeneralProof, GeneralOutput)> {
    let (a0, b0, v_maps) = reduction.group_maps(left, right, om)?;
    let panels = reduction.panel_ends();
    let summand = Summand::PanelProduct {
        lam: reduction.lam,
        panels: panels.clone(),
    };
    let variables = (0..reduction.lam)
        .map(|_| crate::proof::psc::Weight::plain(3))
        .collect::<Result<Vec<_>>>()?;
    let mut inner = GroupInner::new(a0, b0, v_maps, panels, reduction.lam)?;
    prove_general(
        &variables,
        claim_masks,
        mask_value,
        target,
        &summand,
        None,
        &mut inner,
        masks,
        challenges,
    )
}

/// Verifies the stack's whole end group.
pub fn verify_stack_group(
    reduction: &StackReduction,
    claim_masks: &[MaskTerm],
    target: Fp3,
    proof: &GeneralProof,
    hiding_entries: usize,
    challenges: &mut dyn GeneralChallenges,
) -> Result<GeneralOutput> {
    let summand = Summand::PanelProduct {
        lam: reduction.lam,
        panels: reduction.panel_ends(),
    };
    let variables = (0..reduction.lam)
        .map(|_| crate::proof::psc::Weight::plain(3))
        .collect::<Result<Vec<_>>>()?;
    verify_general(
        &variables,
        claim_masks,
        target,
        &summand,
        None,
        proof,
        hiding_entries,
        challenges,
    )
}

/// Proves the stack's batched sumcheck. The summand is the product of the two
/// sides' end values; the stack's variables are plain sums.
#[allow(clippy::too_many_arguments)]
pub fn prove_stack(
    reduction: &StackReduction,
    left: &[Vec<Fp3>],
    right: &[Vec<Fp3>],
    om: &[Fp3],
    claim_masks: &[crate::proof::psc_general::MaskTerm],
    mask_value: &dyn Fn(usize) -> Result<Fp3>,
    shown: Fp3,
    masks: &crate::proof::psc_general::GeneralMaskEntries,
    challenges: &mut dyn crate::proof::psc_general::GeneralChallenges,
) -> Result<(crate::proof::psc_general::GeneralProof, crate::proof::psc_general::GeneralOutput)> {
    let (left_map, right_map) = reduction.side_maps(left, right, om)?;
    let variables = (0..reduction.lam)
        .map(|_| crate::proof::psc::Weight::plain(3))
        .collect::<Result<Vec<_>>>()?;
    let mut inner = crate::proof::psc_general::ProductInner::new(&left_map, &right_map)?;
    crate::proof::psc_general::prove_general(
        &variables,
        claim_masks,
        mask_value,
        shown,
        &crate::proof::psc_general::Summand::Product,
        None,
        &mut inner,
        masks,
        challenges,
    )
}

/// Verifies the stack's batched sumcheck and returns its output claim.
pub fn verify_stack(
    reduction: &StackReduction,
    claim_masks: &[crate::proof::psc_general::MaskTerm],
    shown: Fp3,
    proof: &crate::proof::psc_general::GeneralProof,
    hiding_entries: usize,
    challenges: &mut dyn crate::proof::psc_general::GeneralChallenges,
) -> Result<crate::proof::psc_general::GeneralOutput> {
    let variables = (0..reduction.lam)
        .map(|_| crate::proof::psc::Weight::plain(3))
        .collect::<Result<Vec<_>>>()?;
    crate::proof::psc_general::verify_general(
        &variables,
        claim_masks,
        shown,
        &crate::proof::psc_general::Summand::Product,
        None,
        proof,
        hiding_entries,
        challenges,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(value: u64) -> Fp3 {
        Fp3::from_u64(value)
    }

    #[test]
    fn panels_are_laid_out_largest_first_and_aligned() {
        let mut panels = vec![
            StackPanel {
                claim: 0,
                ec: fp(1),
                ni: 2,
                ny: 2,
                weight: vec![fp(1), fp(2)],
                first_claim: None,
                w: 0,
                off: 0,
            },
            StackPanel {
                claim: 0,
                ec: fp(1),
                ni: 4,
                ny: 2,
                weight: vec![fp(1); 4],
                first_claim: None,
                w: 0,
                off: 0,
            },
        ];
        let lam = panel_layout(&mut panels).unwrap();
        assert_eq!(lam, 4);
        assert_eq!(panels[1].off, 0);
        assert_eq!(panels[1].w, 3);
        assert_eq!(panels[0].off, 8);
        assert_eq!(panels[0].w, 2);
        // A row weight per row and whole powers of two are required.
        let mut bad = vec![StackPanel {
            claim: 0,
            ec: fp(1),
            ni: 2,
            ny: 2,
            weight: vec![fp(1)],
            first_claim: None,
            w: 0,
            off: 0,
        }];
        assert!(panel_layout(&mut bad).is_err());
        let mut bad_rows = vec![StackPanel {
            claim: 0,
            ec: fp(1),
            ni: 3,
            ny: 2,
            weight: vec![fp(1); 3],
            first_claim: None,
            w: 0,
            off: 0,
        }];
        assert!(panel_layout(&mut bad_rows).is_err());
    }

    #[test]
    fn the_affine_terms_recombine_the_panels() {
        // One panel of two rows and two entries (w = 2) at offset 0 of a
        // two-bit stack; the batch coefficient is three and the call factor
        // five.
        let panel = StackPanel {
            claim: 1,
            ec: fp(5),
            ni: 2,
            ny: 2,
            weight: vec![fp(7), fp(11)],
            first_claim: Some(0),
            w: 2,
            off: 0,
        };
        let rs = [fp(2), fp(3)];
        let om = [fp(1), fp(3)];
        let affines = stack_affines(&[panel.clone()], &rs, 2, &om).unwrap();
        // sb of the whole stack is one; the coordinates pair as rs[0] for the
        // entry bit and rs[1] for the row bit.
        let ey = eq_low(&rs, 0, 1).unwrap();
        let ei = eq_low(&rs, 1, 2).unwrap();
        let expected = fp(5)
            .mul(&fp(3))
            .mul(&fp(7))
            .mul(&ei[0])
            .mul(&ey[0]);
        assert_eq!(affines.alin[0].coefficient, expected);
        // The second side is weights: one shown term per row.
        assert!(affines.blin.is_empty());
        assert_eq!(affines.shown.len(), 2);
        assert_eq!(affines.shown[0].2, ei[0]);
        // A point of the wrong width and a missing batch coefficient are
        // refused.
        assert!(stack_affines(&[panel.clone()], &rs[..1], 2, &om).is_err());
        assert!(stack_affines(&[panel], &rs, 2, &[]).is_err());
    }

    #[test]
    fn equality_ranges_pair_bits_low_first() {
        let rs = [fp(2), fp(3), fp(5)];
        let table = eq_low(&rs, 1, 3).unwrap();
        assert_eq!(table.len(), 4);
        // Entry ce pairs bit 0 with rs[1] and bit 1 with rs[2].
        for ce in 0..4u64 {
            let expected = if ce & 1 == 1 { fp(3) } else { Fp3::ONE.sub(&fp(3)) };
            let expected = expected.mul(&if ce & 2 == 2 { fp(5) } else { Fp3::ONE.sub(&fp(5)) });
            assert_eq!(table[ce as usize], expected);
            assert_eq!(eq_low_at(&rs, 1, 3, ce).unwrap(), expected);
        }
        assert!(eq_low(&rs, 2, 2).is_err());
        assert!(eq_low(&rs, 1, 4).is_err());
    }

    use crate::proof::psc::PscRound;
    use crate::proof::psc_general::{
        general_variables, GeneralChallenges, GeneralMaskEntries, Summand,
    };

    struct Source(u64);

    impl GeneralChallenges for Source {
        fn draw(&mut self) -> Result<Fp3> {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            Ok(Fp3::from_u64(self.0))
        }
        fn draw_nonzero(&mut self) -> Result<Fp3> {
            for _ in 0..64 {
                let value = self.draw()?;
                if value != Fp3::ZERO {
                    return Ok(value);
                }
            }
            Err(Error::EntropyUnavailable)
        }
        fn send_hiding_sum(&mut self, _value: Fp3) -> Result<()> {
            Ok(())
        }
        fn send_round(&mut self, _index: usize, _round: &PscRound) -> Result<()> {
            Ok(())
        }
        fn send_end_values(&mut self, _values: &[Fp3]) -> Result<()> {
            Ok(())
        }
    }

    fn panel(claim: usize, ec: u64, ni: u64, ny: u64, weight: Vec<Fp3>) -> StackPanel {
        StackPanel {
            claim,
            ec: fp(ec),
            ni,
            ny,
            weight,
            first_claim: None,
            w: 0,
            off: 0,
        }
    }

    fn cells(count: u64, base: u64) -> Vec<Fp3> {
        (0..count).map(|index| fp(base + index)).collect()
    }

    fn two_panels() -> (StackReduction, Vec<Fp3>, Vec<Fp3>) {
        let small = panel(0, 5, 2, 2, vec![fp(7), fp(11)]);
        let large = panel(1, 3, 4, 4, vec![fp(1), fp(2), fp(3), fp(4)]);
        let reduction = StackReduction::new(vec![small, large]).unwrap();
        let om = vec![fp(2), fp(5)];
        (reduction, om, vec![fp(2), fp(3), fp(4), fp(5), fp(6)])
    }

    #[test]
    fn the_stack_maps_pack_the_panels_at_their_offsets() {
        let (reduction, om, _) = two_panels();
        // The larger panel comes first at offset zero; the smaller one is
        // aligned above it.
        assert_eq!(reduction.lam, 5);
        assert_eq!(reduction.panels[1].off, 0);
        assert_eq!(reduction.panels[0].off, 16);
        let left = vec![cells(4, 1), cells(16, 101)];
        let right = vec![cells(4, 201), cells(16, 301)];
        let (left_map, right_map) = reduction.side_maps(&left, &right, &om).unwrap();
        // The large panel's first row: call factor three, batch five, weight
        // one, entry one.
        assert_eq!(left_map[0], fp(15).mul(&left[1][0]));
        assert_eq!(left_map[1], fp(15).mul(&left[1][1]));
        assert_eq!(left_map[4], fp(30).mul(&left[1][4]));
        // The small panel sits at its own offset with call factor five and
        // batch two.
        assert_eq!(left_map[16], fp(70).mul(&left[0][0]));
        assert_eq!(left_map[19], fp(110).mul(&left[0][3]));
        for index in 0..4 {
            assert_eq!(right_map[16 + index], right[0][index]);
            assert_eq!(right_map[index % 4 + 4 * (index % 4)], right[1][(index % 4) * 4 + index % 4]);
        }
        assert_eq!(right_map[0], right[1][0]);
        assert_eq!(right_map[15], right[1][15]);
        // Only the panels' cells are nonempty.
        assert_eq!(right_map[20], Fp3::ZERO);
        // Shapes and batch coefficients are checked.
        assert!(reduction.side_maps(&left, &right[..1], &om).is_err());
        assert!(reduction.side_maps(&left, &right, &om[..1]).is_err());
    }

    #[test]
    fn the_stack_claim_sums_the_panel_products() {
        let (reduction, om, _) = two_panels();
        let left = vec![cells(4, 1), cells(16, 101)];
        let right = vec![cells(4, 201), cells(16, 301)];
        let mut expected = Fp3::ZERO;
        for (index, panel) in reduction.panels.iter().enumerate() {
            let batch = om[panel.claim];
            for i in 0..panel.ni as usize {
                for yv in 0..panel.ny as usize {
                    let cell = i * panel.ny as usize + yv;
                    expected = expected.add(
                        &panel
                            .ec
                            .mul(&batch)
                            .mul(&panel.weight[i])
                            .mul(&left[index][cell])
                            .mul(&right[index][cell]),
                    );
                }
            }
        }
        assert_eq!(reduction.sum(&left, &right, &om).unwrap(), expected);
    }

    #[test]
    fn the_row_ties_tie_the_end_values_to_the_reads() {
        // One read panel and one weight panel: the weight panel's second
        // side repeats the shown row weight across its entries.
        let read = panel(0, 5, 2, 2, vec![fp(7), fp(11)]);
        let weight = StackPanel {
            first_claim: Some(0),
            ..panel(0, 3, 2, 2, vec![fp(1), fp(1)])
        };
        let reduction = StackReduction::new(vec![read, weight]).unwrap();
        let om = vec![fp(2)];
        let left = vec![cells(4, 1), cells(4, 51)];
        let row_weight = |row: usize| fp(13 + row as u64);
        let right = vec![
            cells(4, 201),
            vec![
                row_weight(0), row_weight(0), row_weight(1), row_weight(1),
            ],
        ];
        let (_, point) = {
            let (reduction_ref, _, point) = two_panels();
            (reduction_ref, point)
        };
        let point = point[..reduction.lam as usize].to_vec();
        let (left_end, right_end) = reduction.end_values(&left, &right, &om, &point).unwrap();
        reduction
            .check_row_ties(&left, &right, &om, &point, left_end, right_end)
            .unwrap();
        // A changed read moves the recomposed value away from the sent end
        // value.
        let mut changed = left.clone();
        changed[0][2] = changed[0][2].add(&Fp3::ONE);
        assert!(reduction
            .check_row_ties(&changed, &right, &om, &point, left_end, right_end)
            .is_err());
        // So does a changed second-side read or a changed shown weight.
        let mut changed = right.clone();
        changed[0][1] = changed[0][1].add(&Fp3::ONE);
        assert!(reduction
            .check_row_ties(&left, &changed, &om, &point, left_end, right_end)
            .is_err());
        let mut changed = right.clone();
        changed[1][0] = changed[1][0].add(&Fp3::ONE);
        assert!(reduction
            .check_row_ties(&left, &changed, &om, &point, left_end, right_end)
            .is_err());
    }

    #[test]
    fn the_group_ties_bind_the_rest_and_weight_values() {
        let read = panel(0, 5, 2, 2, vec![fp(7), fp(11)]);
        let weight = StackPanel {
            first_claim: Some(0),
            ..panel(0, 3, 2, 2, vec![fp(1), fp(1)])
        };
        let reduction = StackReduction::new(vec![read, weight]).unwrap();
        let om = vec![fp(2)];
        let point = vec![fp(3), fp(4), fp(5)];
        let left = vec![cells(4, 1), cells(4, 51)];
        let mut right = vec![cells(4, 201), vec![Fp3::ZERO; 4]];
        let factor = fp(9);
        let mask = |slot: usize| Ok(fp(700 + slot as u64));
        let affines = stack_affines(&reduction.panels, &point, reduction.lam, &om).unwrap();
        let mut alin = Fp3::ZERO;
        for term in &affines.alin {
            let panel = &reduction.panels[term.panel];
            alin = alin.add(
                &term
                    .coefficient
                    .mul(&left[term.panel][(term.i * panel.ny + term.yv) as usize]),
            );
        }
        let mut blin = Fp3::ZERO;
        for term in &affines.blin {
            let panel = &reduction.panels[term.panel];
            blin = blin.add(
                &term
                    .coefficient
                    .mul(&right[term.panel][(term.i * panel.ny + term.yv) as usize]),
            );
        }
        let v = vec![fp(13), fp(17)];
        let mut sent = vec![alin, blin];
        sent.extend(&v);
        let sent: Vec<Fp3> = sent
            .iter()
            .enumerate()
            .map(|(slot, value)| value.add(&factor.mul(&mask(slot).unwrap())))
            .collect();
        let slots: Vec<usize> = (0..sent.len()).collect();
        let group = GroupValues::from_sent(&sent, &slots, factor, 0, &mask).unwrap();
        assert_eq!(group.a0, alin);
        assert_eq!(group.b0, blin);
        assert_eq!(group.v, v);
        let (as_, bs) = reduction
            .group_bases(&left, &right, &om, &point, &group)
            .unwrap();
        assert_eq!(as_, alin);
        let mut expected = blin;
        for ((_, _, coefficient), value) in affines.shown.iter().zip(&v) {
            expected = expected.add(&coefficient.mul(value));
        }
        assert_eq!(bs, expected);
        // Changed reads or a short group break the ties.
        let mut changed = left.clone();
        changed[0][1] = changed[0][1].add(&Fp3::ONE);
        assert!(reduction
            .group_bases(&changed, &right, &om, &point, &group)
            .is_err());
        right[0][1] = right[0][1].add(&Fp3::ONE);
        assert!(reduction
            .group_bases(&left, &right, &om, &point, &group)
            .is_err());
        let mut short = group.clone();
        short.v.pop();
        assert!(reduction
            .group_bases(&left, &right, &om, &point, &short)
            .is_err());
    }

    #[test]
    fn the_end_ties_bind_the_hidden_end_values_to_the_panels() {
        // One read panel and one weight panel.
        let read = panel(0, 5, 2, 2, vec![fp(7), fp(11)]);
        let weight = StackPanel {
            first_claim: Some(0),
            ..panel(0, 3, 2, 2, vec![fp(1), fp(1)])
        };
        let reduction = StackReduction::new(vec![read, weight]).unwrap();
        let om = vec![fp(2)];
        // Two panels of four cells fill a three-bit stack.
        let point = vec![fp(3), fp(4), fp(5)];
        let left = vec![cells(4, 1), cells(4, 51)];
        let right = vec![cells(4, 201), cells(4, 301)];
        let scale = fp(9);
        let mask = |slot: usize| Ok(fp(700 + slot as u64));
        let affines = stack_affines(&reduction.panels, &point, reduction.lam, &om).unwrap();
        let mut alin = Fp3::ZERO;
        for term in &affines.alin {
            let panel = &reduction.panels[term.panel];
            alin = alin.add(
                &term
                    .coefficient
                    .mul(&left[term.panel][(term.i * panel.ny + term.yv) as usize]),
            );
        }
        let mut blin = Fp3::ZERO;
        for term in &affines.blin {
            let panel = &reduction.panels[term.panel];
            blin = blin.add(
                &term
                    .coefficient
                    .mul(&right[term.panel][(term.i * panel.ny + term.yv) as usize]),
            );
        }
        for (panel_index, i, coefficient) in &affines.shown {
            let panel = &reduction.panels[*panel_index];
            blin = blin.add(&coefficient.mul(&right[*panel_index][(*i * panel.ny) as usize]));
        }
        let left_end = EndValue {
            value: alin.add(&scale.mul(&mask(0).unwrap())),
            slot: 0,
            scale,
        };
        let right_end = EndValue {
            value: blin.add(&scale.mul(&mask(1).unwrap())),
            slot: 1,
            scale,
        };
        reduction
            .check_end_ties(&left, &right, &om, &point, left_end, right_end, &mask)
            .unwrap();
        // A changed read, shown weight, end value or mask entry is refused.
        let mut changed = left.clone();
        changed[0][1] = changed[0][1].add(&Fp3::ONE);
        assert!(reduction
            .check_end_ties(&changed, &right, &om, &point, left_end, right_end, &mask)
            .is_err());
        let mut changed = right.clone();
        changed[1][0] = changed[1][0].add(&Fp3::ONE);
        assert!(reduction
            .check_end_ties(&left, &changed, &om, &point, left_end, right_end, &mask)
            .is_err());
        let mut changed = right_end;
        changed.value = changed.value.add(&Fp3::ONE);
        assert!(reduction
            .check_end_ties(&left, &right, &om, &point, left_end, changed, &mask)
            .is_err());
        let other_mask = |slot: usize| Ok(fp(701 + slot as u64));
        assert!(reduction
            .check_end_ties(&left, &right, &om, &point, left_end, right_end, &other_mask)
            .is_err());
    }

    #[test]
    fn the_stack_sumcheck_proves_and_verifies_its_claim() {
        let read = panel(0, 5, 2, 2, vec![fp(7), fp(11)]);
        let reduction = StackReduction::new(vec![read]).unwrap();
        let om = vec![fp(2), fp(3)];
        let left = vec![cells(4, 1)];
        let right = vec![cells(4, 201)];
        let claim = reduction.sum(&left, &right, &om).unwrap();
        let variables = (0..reduction.lam)
            .map(|_| crate::proof::psc::Weight::plain(3))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let hiding_count = crate::proof::psc::hp_entries(&general_variables(
            &variables,
            &Summand::Product,
        )
        .unwrap())
        .unwrap();
        let masks = GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| fp(700 + index as u64)).collect(),
            shown: vec![fp(800), fp(801)],
        };
        let (proof, proven) = prove_stack(
            &reduction,
            &left,
            &right,
            &om,
            &[],
            &|_| Ok(Fp3::ZERO),
            claim,
            &masks,
            &mut Source(41),
        )
        .unwrap();
        let verified = verify_stack(
            &reduction,
            &[],
            claim,
            &proof,
            masks.hiding.len(),
            &mut Source(41),
        )
        .unwrap();
        assert_eq!(proven, verified);
        assert_eq!(proven.point.len(), reduction.lam as usize);
        // A changed round leaves another output claim.
        let mut changed = proof.clone();
        changed.rounds[0].coefficients[0] =
            changed.rounds[0].coefficients[0].add(&Fp3::ONE);
        let tampered = verify_stack(
            &reduction,
            &[],
            claim,
            &changed,
            masks.hiding.len(),
            &mut Source(41),
        )
        .unwrap();
        assert_ne!(tampered, verified);
        // Another target does not match the first round.
        assert!(prove_stack(
            &reduction,
            &left,
            &right,
            &om,
            &[],
            &|_| Ok(Fp3::ZERO),
            claim.add(&Fp3::ONE),
            &masks,
            &mut Source(41),
        )
        .is_err());
    }

    #[test]
    fn the_group_stack_entry_proves_and_verifies() {
        let read = panel(0, 5, 2, 2, vec![fp(7), fp(11)]);
        let weight = StackPanel {
            first_claim: Some(0),
            ..panel(0, 3, 2, 2, vec![fp(1), fp(1)])
        };
        let reduction = StackReduction::new(vec![read, weight]).unwrap();
        let om = vec![fp(2)];
        let left = vec![cells(4, 1), cells(4, 51)];
        let right = vec![cells(4, 201), cells(4, 301)];
        let (a0, b0, v_maps) = reduction.group_maps(&left, &right, &om).unwrap();
        assert_eq!(v_maps.len(), 2);
        let mut packed_right = b0.clone();
        for map in &v_maps {
            for (slot, value) in packed_right.iter_mut().zip(map) {
                *slot = slot.add(value);
            }
        }
        let mut target = Fp3::ZERO;
        for (left_value, right_value) in a0.iter().zip(&packed_right) {
            target = target.add(&left_value.mul(right_value));
        }
        let variables = (0..reduction.lam)
            .map(|_| crate::proof::psc::Weight::plain(3))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let panels = reduction.panel_ends();
        let summand = Summand::PanelProduct {
            lam: reduction.lam,
            panels: panels.clone(),
        };
        let hiding_count = crate::proof::psc::hp_entries(
            &crate::proof::psc_general::general_variables(&variables, &summand).unwrap(),
        )
        .unwrap();
        let shown = summand.shown();
        assert_eq!(shown, 4);
        let masks = GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| fp(700 + index as u64)).collect(),
            shown: (0..shown).map(|index| fp(800 + index as u64)).collect(),
        };
        let (proof, proven) = prove_stack_group(
            &reduction,
            &left,
            &right,
            &om,
            &[],
            &|_| Ok(Fp3::ZERO),
            target,
            &masks,
            &mut Source(71),
        )
        .unwrap();
        let verified = verify_stack_group(
            &reduction,
            &[],
            target,
            &proof,
            masks.hiding.len(),
            &mut Source(71),
        )
        .unwrap();
        assert_eq!(proven, verified);
        assert_eq!(proven.ends.len(), shown);
        let beta = proven.ends[0].mask_scale;
        let mut hidden = Vec::with_capacity(shown);
        for (end, mask) in proven.ends.iter().zip(&masks.shown) {
            hidden.push(end.value.sub(&beta.mul(mask)));
        }
        let group = GroupValues {
            a0: hidden[0],
            b0: hidden[1],
            v: hidden[2..].to_vec(),
        };
        let (as_, bs) = reduction
            .group_bases(&left, &right, &om, &proven.point, &group)
            .unwrap();
        assert_eq!(as_, hidden[0]);
        // Another last round leaves another point or group.
        let mut changed = proof.clone();
        changed.y_round.coefficients[0] = changed.y_round.coefficients[0].add(&Fp3::ONE);
        let other = verify_stack_group(
            &reduction,
            &[],
            target,
            &changed,
            masks.hiding.len(),
            &mut Source(71),
        )
        .unwrap();
        assert_ne!(other, verified);
        // Another weight value does not sum to the claim.
        let mut changed_right = right.clone();
        changed_right[1][0] = changed_right[1][0].add(&Fp3::ONE);
        assert!(prove_stack_group(
            &reduction,
            &left,
            &changed_right,
            &om,
            &[],
            &|_| Ok(Fp3::ZERO),
            target,
            &masks,
            &mut Source(71),
        )
        .is_err());
        // The terminal second side is the packed B side at the point.
        assert_eq!(
            bs,
            reduction.mle(&packed_right, &proven.point).unwrap()
        );
    }
}
