#![allow(incomplete_features)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]

pub mod activations;
pub mod layers;

// Trait to use with `derive` on structs containing the neural network layers
pub trait NeuralNetwork<const I: usize, const O: usize> {
    fn forward(&self, input: [f32; I]) -> [f32; O];
}

#[cfg(test)]
mod tests {
    use super::NeuralNetwork;
    use crate::activations::relu;
    use crate::layers::Dense;
    use gamma_derive::NeuralNetwork;

    use rand::{distributions::Uniform, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[derive(NeuralNetwork)]
    struct TestNetwork {
        input: Dense<2, 3>,
        hidden: Dense<3, 3>,
        output: Dense<3, 1>,
    }

    #[test]
    fn network_forward() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let dist = Uniform::from(-1.0..=1.0);

        let nn = TestNetwork {
            input: Dense::random(&mut rng, &dist, relu),
            hidden: Dense::random(&mut rng, &dist, relu),
            output: Dense::random(&mut rng, &dist, relu),
        };

        let output = nn.forward([1.0, 1.0]);
        assert!((output[0] - 0.0).abs() < f32::EPSILON);
    }
}
