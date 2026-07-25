//! ONNX model parsing: decode `.onnx` protobuf files into typed structures,
//! extract initializer weights, and perform best-effort static shape lookup.
//!
//! # Scope limitations (deliberate)
//!
//! - Only `TensorProto.data_type == FLOAT` (1) is supported for initializer
//!   extraction. Other numeric dtypes (INT64, DOUBLE, BOOL, quantized 8/4/2
//!   bit formats, etc.) are out of scope for this stage of the compiler and
//!   produce a typed error rather than being silently misinterpreted.
//! - Converting weights to `I18` fixed point (see [`weight_to_i18`]) WILL
//!   fail for any f32 magnitude outside `I18`'s representable range
//!   (~±9.22). This is a real, documented limitation of the current
//!   foundation-layer fixed-point format: many realistic model weights
//!   (embeddings, learned scales, etc.) exceed this range, and this
//!   function surfaces that as a typed error rather than panicking or
//!   wrapping/saturating silently.
//! - Shape extraction ([`extract_value_shapes`]) is best-effort: ONNX
//!   allows symbolic/dynamic dimensions (`dim_param`, e.g. a symbolic batch
//!   size). If ANY dimension of a value's shape is symbolic or otherwise
//!   unresolvable, the ENTIRE shape for that value is omitted from the
//!   returned map — we never emit a partially-resolved shape, since
//!   downstream axis-indexed consumers (e.g. the op mapper's `m`/`n`/`k`
//!   extraction) would silently misinterpret a shape that is missing a
//!   dimension from the middle. Full shape inference for graphs with
//!   dynamic axes is out of scope; this compiler works best on graphs
//!   exported with fully static shapes.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::Path;

use prost::Message;

use crate::onnx::{
    tensor_shape_proto, type_proto, GraphProto, ModelProto, TensorProto, ValueInfoProto,
};
use zkie_core::fixed_point::{FixedPointError, I18};

/// ONNX `TensorProto.DataType::FLOAT` — the only element type this
/// foundation-layer compiler currently understands. See module docs.
const ONNX_DATA_TYPE_FLOAT: i32 = 1;

/// Errors that can occur while loading an ONNX model or extracting its
/// weights/shapes.
#[derive(Debug)]
pub enum OnnxParseError {
    /// The model file could not be read from disk.
    Io(io::Error),
    /// The file's bytes were not a valid `ModelProto` protobuf message.
    Decode(prost::DecodeError),
    /// A tensor's `dims` contained a negative value, which is not a valid
    /// shape dimension.
    NegativeDimension { tensor_name: String, dim: i64 },
    /// A tensor's `data_type` was not `FLOAT` (1). See module docs.
    UnsupportedDataType { tensor_name: String, data_type: i32 },
    /// A tensor's `raw_data` length was not a multiple of 4 bytes, so it
    /// cannot be reinterpreted as a sequence of `f32` values.
    RawDataLengthMismatch { tensor_name: String, len: usize },
    /// A weight value was outside `I18`'s representable range during
    /// fixed-point conversion.
    WeightOutOfRange {
        index: usize,
        value: f32,
        source: FixedPointError,
    },
    /// A requested weight name was not present in an extracted initializer
    /// map. This does not indicate a malformed ONNX model — it means a
    /// caller (e.g. `graph_compiler::CompiledProgram::weight_as_i18`) asked
    /// for a weight tensor name that doesn't exist among the graph's
    /// initializers.
    WeightNotFound(String),
}

impl fmt::Display for OnnxParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxParseError::Io(e) => write!(f, "failed to read ONNX model file: {e}"),
            OnnxParseError::Decode(e) => write!(f, "failed to decode ONNX protobuf: {e}"),
            OnnxParseError::NegativeDimension { tensor_name, dim } => write!(
                f,
                "tensor '{tensor_name}' has a negative dimension ({dim}), which is not a valid shape"
            ),
            OnnxParseError::UnsupportedDataType {
                tensor_name,
                data_type,
            } => write!(
                f,
                "tensor '{tensor_name}' has unsupported data_type {data_type}; only FLOAT (1) is supported by this compiler"
            ),
            OnnxParseError::RawDataLengthMismatch { tensor_name, len } => write!(
                f,
                "tensor '{tensor_name}' has raw_data of length {len} bytes, which is not a multiple of 4 (f32 size)"
            ),
            OnnxParseError::WeightOutOfRange {
                index,
                value,
                source,
            } => write!(
                f,
                "weight value {value} at index {index} does not fit in I18 fixed-point range: {source}"
            ),
            OnnxParseError::WeightNotFound(name) => write!(
                f,
                "no weight named '{name}' was found among this model's initializers"
            ),
        }
    }
}

impl std::error::Error for OnnxParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OnnxParseError::Io(e) => Some(e),
            OnnxParseError::Decode(e) => Some(e),
            OnnxParseError::WeightOutOfRange { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<io::Error> for OnnxParseError {
    fn from(e: io::Error) -> Self {
        OnnxParseError::Io(e)
    }
}

impl From<prost::DecodeError> for OnnxParseError {
    fn from(e: prost::DecodeError) -> Self {
        OnnxParseError::Decode(e)
    }
}

/// Reads the file at `path` and decodes it as an ONNX `ModelProto`.
pub fn load_model(path: impl AsRef<Path>) -> Result<ModelProto, OnnxParseError> {
    let bytes = std::fs::read(path)?;
    let model = ModelProto::decode(bytes.as_slice())?;
    Ok(model)
}

/// A dense weight tensor extracted from an ONNX `TensorProto` initializer:
/// its row-major shape plus its values as `f32` (ONNX's native float
/// encoding, prior to any fixed-point conversion).
#[derive(Debug, Clone, PartialEq)]
pub struct WeightTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Extracts every initializer tensor in `graph` into a name-keyed map of
/// [`WeightTensor`]s. See module docs for the FLOAT-only scope limitation.
pub fn extract_initializers(
    graph: &GraphProto,
) -> Result<HashMap<String, WeightTensor>, OnnxParseError> {
    graph
        .initializer
        .iter()
        .map(|tensor| Ok((tensor.name.clone(), extract_tensor(tensor)?)))
        .collect()
}

fn extract_tensor(tensor: &TensorProto) -> Result<WeightTensor, OnnxParseError> {
    let shape = tensor
        .dims
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| OnnxParseError::NegativeDimension {
                tensor_name: tensor.name.clone(),
                dim,
            })
        })
        .collect::<Result<Vec<usize>, _>>()?;

    if tensor.data_type != ONNX_DATA_TYPE_FLOAT {
        return Err(OnnxParseError::UnsupportedDataType {
            tensor_name: tensor.name.clone(),
            data_type: tensor.data_type,
        });
    }

    let data = if !tensor.float_data.is_empty() {
        tensor.float_data.clone()
    } else if !tensor.raw_data.is_empty() {
        if !tensor.raw_data.len().is_multiple_of(4) {
            return Err(OnnxParseError::RawDataLengthMismatch {
                tensor_name: tensor.name.clone(),
                len: tensor.raw_data.len(),
            });
        }
        tensor
            .raw_data
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact(4) yields len-4 slices"),
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(WeightTensor { shape, data })
}

/// Converts a [`WeightTensor`]'s `f32` values into `I18` fixed-point, in
/// order. Propagates the first out-of-range value as a typed error rather
/// than panicking — see module docs: this WILL happen for realistic model
/// weights outside I18's tiny representable range, and that is a real,
/// documented limitation of this foundation-layer compiler.
pub fn weight_to_i18(tensor: &WeightTensor) -> Result<Vec<I18>, OnnxParseError> {
    tensor
        .data
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            I18::from_f64(value as f64).map_err(|source| OnnxParseError::WeightOutOfRange {
                index,
                value,
                source,
            })
        })
        .collect()
}

/// Best-effort static shape lookup for every named value in `graph.input`,
/// `graph.output`, and `graph.value_info`. See module docs for the
/// symbolic-dimension scope limitation: a value with any dynamic dimension
/// is simply absent from the returned map.
pub fn extract_value_shapes(graph: &GraphProto) -> HashMap<String, Vec<usize>> {
    graph
        .input
        .iter()
        .chain(graph.output.iter())
        .chain(graph.value_info.iter())
        .filter_map(|value_info| {
            static_shape_of(value_info).map(|shape| (value_info.name.clone(), shape))
        })
        .collect()
}

fn static_shape_of(value_info: &ValueInfoProto) -> Option<Vec<usize>> {
    let type_proto = value_info.r#type.as_ref()?;
    let tensor_type = match type_proto.value.as_ref()? {
        type_proto::Value::TensorType(t) => t,
        _ => return None,
    };
    let shape_proto = tensor_type.shape.as_ref()?;
    shape_proto
        .dim
        .iter()
        .map(|dim| match dim.value.as_ref() {
            Some(tensor_shape_proto::dimension::Value::DimValue(v)) if *v >= 0 => Some(*v as usize),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::{NodeProto, TensorShapeProto};

    fn write_temp_file(bytes: &[u8], suffix: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "zkie_compiler_test_{}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            suffix
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn load_model_decodes_valid_bytes() {
        let model = ModelProto {
            ir_version: 9,
            graph: Some(GraphProto {
                name: "g".to_string(),
                node: vec![NodeProto {
                    op_type: "Relu".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let bytes = model.encode_to_vec();
        let path = write_temp_file(&bytes, "valid.onnx");

        let loaded = load_model(&path).expect("should decode");
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.ir_version, 9);
        assert_eq!(loaded.graph.unwrap().node.len(), 1);
    }

    #[test]
    fn load_model_returns_io_error_for_missing_file() {
        let result = load_model("/nonexistent/path/does/not/exist.onnx");
        match result {
            Err(OnnxParseError::Io(_)) => {}
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    #[test]
    fn load_model_returns_decode_error_for_malformed_bytes() {
        let path = write_temp_file(&[0xFF, 0x00, 0xAB, 0xCD, 0xEF], "malformed.onnx");
        let result = load_model(&path);
        std::fs::remove_file(&path).ok();
        match result {
            Err(OnnxParseError::Decode(_)) => {}
            other => panic!("expected Decode error, got {other:?}"),
        }
    }

    #[test]
    fn extract_initializers_reads_float_data() {
        let graph = GraphProto {
            initializer: vec![TensorProto {
                name: "w".to_string(),
                dims: vec![2, 2],
                data_type: ONNX_DATA_TYPE_FLOAT,
                float_data: vec![1.0, 2.0, 3.0, 4.0],
                ..Default::default()
            }],
            ..Default::default()
        };

        let weights = extract_initializers(&graph).unwrap();
        let w = weights.get("w").unwrap();
        assert_eq!(w.shape, vec![2, 2]);
        assert_eq!(w.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn extract_initializers_reads_raw_data_as_little_endian_f32() {
        let values = [1.0f32, -2.5, 3.25, 4.0];
        let mut raw = Vec::new();
        for v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let graph = GraphProto {
            initializer: vec![TensorProto {
                name: "w".to_string(),
                dims: vec![4],
                data_type: ONNX_DATA_TYPE_FLOAT,
                raw_data: raw,
                ..Default::default()
            }],
            ..Default::default()
        };

        let weights = extract_initializers(&graph).unwrap();
        let w = weights.get("w").unwrap();
        assert_eq!(w.shape, vec![4]);
        assert_eq!(w.data, values.to_vec());
    }

    #[test]
    fn extract_initializers_rejects_negative_dimension() {
        let graph = GraphProto {
            initializer: vec![TensorProto {
                name: "bad".to_string(),
                dims: vec![-1, 2],
                data_type: ONNX_DATA_TYPE_FLOAT,
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = extract_initializers(&graph);
        match result {
            Err(OnnxParseError::NegativeDimension { tensor_name, dim }) => {
                assert_eq!(tensor_name, "bad");
                assert_eq!(dim, -1);
            }
            other => panic!("expected NegativeDimension error, got {other:?}"),
        }
    }

    #[test]
    fn extract_initializers_rejects_unsupported_data_type() {
        // INT64 == 7
        let graph = GraphProto {
            initializer: vec![TensorProto {
                name: "ints".to_string(),
                dims: vec![2],
                data_type: 7,
                int64_data: vec![1, 2],
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = extract_initializers(&graph);
        match result {
            Err(OnnxParseError::UnsupportedDataType {
                tensor_name,
                data_type,
            }) => {
                assert_eq!(tensor_name, "ints");
                assert_eq!(data_type, 7);
            }
            other => panic!("expected UnsupportedDataType error, got {other:?}"),
        }
    }

    #[test]
    fn weight_to_i18_converts_small_values() {
        let tensor = WeightTensor {
            shape: vec![2],
            data: vec![1.5, -2.25],
        };
        let converted = weight_to_i18(&tensor).unwrap();
        assert_eq!(converted.len(), 2);
        assert!((converted[0].to_f64() - 1.5).abs() < 1e-9);
        assert!((converted[1].to_f64() - (-2.25)).abs() < 1e-9);
    }

    #[test]
    fn weight_to_i18_returns_typed_error_for_out_of_range_value() {
        let tensor = WeightTensor {
            shape: vec![1],
            data: vec![1000.0],
        };
        let result = weight_to_i18(&tensor);
        match result {
            Err(OnnxParseError::WeightOutOfRange { index, value, .. }) => {
                assert_eq!(index, 0);
                assert_eq!(value, 1000.0);
            }
            other => panic!("expected WeightOutOfRange error, got {other:?}"),
        }
    }

    fn value_info_with_static_shape(name: &str, dims: Vec<i64>) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(crate::onnx::TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: ONNX_DATA_TYPE_FLOAT,
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .into_iter()
                            .map(|d| tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn extract_value_shapes_reads_static_shapes_from_input_and_output() {
        let graph = GraphProto {
            input: vec![value_info_with_static_shape("x", vec![1, 128])],
            output: vec![value_info_with_static_shape("y", vec![1, 64])],
            ..Default::default()
        };

        let shapes = extract_value_shapes(&graph);
        assert_eq!(shapes.get("x"), Some(&vec![1usize, 128]));
        assert_eq!(shapes.get("y"), Some(&vec![1usize, 64]));
    }

    #[test]
    fn extract_value_shapes_omits_values_with_symbolic_dimensions() {
        let value_info = ValueInfoProto {
            name: "batched".to_string(),
            r#type: Some(crate::onnx::TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: ONNX_DATA_TYPE_FLOAT,
                    shape: Some(TensorShapeProto {
                        dim: vec![
                            tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimParam(
                                    "batch".to_string(),
                                )),
                                ..Default::default()
                            },
                            tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimValue(128)),
                                ..Default::default()
                            },
                        ],
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        };
        let graph = GraphProto {
            value_info: vec![value_info],
            ..Default::default()
        };

        let shapes = extract_value_shapes(&graph);
        assert!(!shapes.contains_key("batched"));
    }
}
