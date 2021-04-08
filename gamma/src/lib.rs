pub mod activations;
pub mod layers;

// Trait to use with `derive` on structs containing the neural network layers
pub trait NeuralNetwork<const I: usize, const O: usize> {
    fn forward(&self, input: [f32; I]) -> [f32; O];
}

#[cfg(test)]
mod tests {
    use super::NeuralNetwork;
    use crate::activations::ReLu;
    use crate::layers::Dense;
    use gamma_derive::NeuralNetwork;

    use rand::{distributions::Uniform, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[derive(NeuralNetwork)]
    struct TestNetwork {
        input: Dense<ReLu, 2, 3>,
        hidden: Dense<ReLu, 3, 3>,
        output: Dense<ReLu, 3, 1>,
    }

    #[test]
    fn network_forward() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let dist = Uniform::from(-1.0..=1.0);

        let nn = TestNetwork {
            input: Dense::random(&mut rng, &dist),
            hidden: Dense::random(&mut rng, &dist),
            output: Dense::random(&mut rng, &dist),
        };

        let output = nn.forward([1.0, 1.0]);
        approx::assert_relative_eq!(output[..], [0.0]);
    }
}
