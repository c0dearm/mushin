pub(crate) type Activation = fn(f32) -> f32;

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::relu;

    #[test]
    fn relu_output() {
        assert!((relu(-1.0) - 0.0).abs() < f32::EPSILON);
        assert!((relu(1.0) - 1.0).abs() < f32::EPSILON);
    }
}
