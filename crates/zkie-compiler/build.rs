fn main() {
    std::env::set_var("PROTOC", protobuf_src::protoc());
    prost_build::compile_protos(&["protos/onnx.proto3"], &["protos/"])
        .expect("failed to compile onnx.proto3");
}
