#[allow(clippy::all)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod graph_compiler;
pub mod onnx_parser;
pub mod op_mapper;

#[cfg(test)]
mod tests {
    use super::onnx::ModelProto;

    #[test]
    fn crate_compiles_and_onnx_types_are_reachable() {
        let model = ModelProto::default();
        assert_eq!(model.ir_version, 0);
    }
}
