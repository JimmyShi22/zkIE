pub mod assembler;
pub mod chip;
pub mod chips;
pub mod field_convert;
pub mod fixed_point;
pub mod isa;
pub mod tensor;

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {
        assert_eq!(2 + 2, 4);
    }
}
