use crate::field_convert::Fr;
use halo2_proofs::circuit::Layouter;
use halo2_proofs::plonk::ErrorFront;

/// Shared vocabulary for chip implementations. `EltwiseAddChip`/`EltwiseMulChip`
/// (in `chips::eltwise`) predate this trait and take two `I18` operands
/// directly rather than a generic `Input`; retrofitting them is left for
/// sub-project 2, where the remaining ISA chips will implement this trait
/// directly instead.
pub trait Chip {
    type Input;

    fn assign(&self, layouter: impl Layouter<Fr>, input: Self::Input) -> Result<(), ErrorFront>;
}
