#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EltwiseOp {
    Add,
    Mul,
    Relu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    DotGeneral {
        m: usize,
        n: usize,
        k: usize,
        batch_dims: Vec<usize>,
        trans_a: bool,
        trans_b: bool,
    },
    Softmax {
        axis_dim: usize,
    },
    Gelu,
    LayerNorm {
        dim: usize,
        epsilon_milli: u64,
    },
    /// RMS normalization (TimesFM's `RMSNorm`, `add_unit_offset=False`):
    /// `output_i = x_i * rsqrt(mean(x^2) + epsilon) * weight_i`. See
    /// `crate::chips::rms_norm::RmsNormChip`. Unlike [`Instruction::LayerNorm`],
    /// this takes a second `RegisterRef` input (`weight`, one value per
    /// channel) alongside the normalized values -- see
    /// `crate::assembler::AssemblerChip`'s `RmsNorm` dispatch.
    RmsNorm {
        dim: usize,
        epsilon_milli: u64,
    },
    Eltwise {
        op: EltwiseOp,
    },
    Reduce {
        op: ReduceOp,
        axis: usize,
    },
    EmbedLookup {
        table_size: usize,
        embed_dim: usize,
    },
    PatchEmbed {
        patch_len: usize,
        embed_dim: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eltwise_add_variant_constructs() {
        let instr = Instruction::Eltwise { op: EltwiseOp::Add };
        assert!(matches!(instr, Instruction::Eltwise { op: EltwiseOp::Add }));
    }

    #[test]
    fn dot_general_variant_constructs() {
        let instr = Instruction::DotGeneral {
            m: 4,
            n: 4,
            k: 4,
            batch_dims: vec![],
            trans_a: false,
            trans_b: true,
        };
        assert!(matches!(instr, Instruction::DotGeneral { .. }));
    }
}
