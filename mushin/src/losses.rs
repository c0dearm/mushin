pub trait Loss<const O: usize> {
    fn loss(output: [f32; O], target: [f32; O]) -> f32;
}

pub struct MeanSquaredError;
pub struct CrossEntropy;

impl<const O: usize> Loss<O> for MeanSquaredError {
    fn loss(output: [f32; O], target: [f32; O]) -> f32 {
        output
            .iter()
            .zip(target.iter())
            .fold(0.0, |acc, (o, t)| acc + (o - t).powi(2))
            / O as f32
    }
}

impl<const O: usize> Loss<O> for CrossEntropy {
    fn loss(output: [f32; O], target: [f32; O]) -> f32 {
        let sum = output.iter().map(|o| o.exp()).sum::<f32>();

        output
            .iter()
            .zip(target.iter())
            .fold(0.0, |acc, (&o, &t)| (sum.ln() - o).mul_add(t, acc))
    }
}

#[cfg(test)]
mod tests {
    use super::Loss;
    use super::{CrossEntropy, MeanSquaredError};

    #[test]
    fn mean_squared_error_loss() {
        approx::assert_relative_eq!(
            MeanSquaredError::loss([1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
            0.5
        );
    }

    #[test]
    fn cross_entropy_loss() {
        approx::assert_relative_eq!(
            CrossEntropy::loss(
                [0.05, 0.95, 0.0, 0.1, 0.8, 0.1],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            ),
            3.360576
        );
    }
}
