pub(crate) type Activation = fn(f32) -> f32;

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::relu;

    #[test]
    fn relu_output() {
        approx::assert_relative_eq!(relu(-1.0), 0.0);
        approx::assert_relative_eq!(relu(1.0), 1.0);
    }
}
