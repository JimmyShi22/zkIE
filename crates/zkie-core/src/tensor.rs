#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> Tensor<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected
            ));
        }
        Ok(Tensor { shape, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_matching_shape_and_data_len() {
        let t = Tensor::new(vec![2, 3], vec![0i64; 6]).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
    }

    #[test]
    fn new_rejects_mismatched_shape_and_data_len() {
        assert!(Tensor::new(vec![2, 3], vec![0i64; 5]).is_err());
    }
}
