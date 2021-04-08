use crate::activations::Activation;
use rand::{distributions::Distribution, Rng, RngCore};

#[derive(Debug)]
pub struct Dense<const I: usize, const O: usize> {
    weights: [[f32; I]; O],
    bias: [f32; O],
    activation: Activation,
}

impl<const I: usize, const O: usize> Dense<I, O> {
    pub fn new(weights: [[f32; I]; O], bias: [f32; O], activation: Activation) -> Self {
        Dense {
            weights,
            bias,
            activation,
        }
    }

    pub fn random<R: RngCore, D: Distribution<f32>>(
        rng: &mut R,
        dist: &D,
        activation: Activation,
    ) -> Self {
        let mut weights = [[0.0; I]; O];
        let mut bias = [0.0; O];

        weights
            .iter_mut()
            .flatten()
            .chain(bias.iter_mut())
            .zip(rng.sample_iter(dist))
            .for_each(|(w, r)| *w = r);

        Dense::new(weights, bias, activation)
    }

    pub fn forward(&self, input: [f32; I]) -> [f32; O] {
        let mut output = [0.0; O];

        output
            .iter_mut()
            .zip(self.bias.iter())
            .enumerate()
            .for_each(|(k, (o, b))| {
                *o = (self.activation)(
                    input
                        .iter()
                        .zip(self.weights[k].iter())
                        .fold(*b, |acc, (x, y)| acc + x * y),
                )
            });

        output
    }
}

#[cfg(test)]
mod tests {
    use super::Dense;
    use crate::activations::relu;

    use rand::{distributions::Uniform, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn dense_new() {
        let layer = Dense::<2, 2>::new([[0.0, 1.0], [2.0, 3.0]], [1.0, 1.0], relu);
        approx::assert_relative_eq!(layer.weights[0][..], [0.0, 1.0]);
        approx::assert_relative_eq!(layer.weights[1][..], [2.0, 3.0]);
        approx::assert_relative_eq!(layer.bias[..], [1.0, 1.0]);
    }

    #[test]
    fn dense_random() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let between = Uniform::from(-1.0..=1.0);
        let layer = Dense::<2, 2>::random(&mut rng, &between, relu);
        approx::assert_relative_eq!(layer.weights[0][..], [-0.6255188, 0.67383957]);
        approx::assert_relative_eq!(layer.weights[1][..], [0.8181262, 0.26284897]);
        approx::assert_relative_eq!(layer.bias[..], [0.5238807, -0.53516835]);
    }

    #[test]
    fn dense_forward() {
        let layer = Dense::<2, 4>::new(
            [[1.0, 1.0], [1.0, 1.0], [-2.0, -2.0], [-2.0, -2.0]],
            [-2.0, 1.0, -2.0, 1.0],
            relu,
        );
        let output = layer.forward([1.0, 1.0]);
        approx::assert_relative_eq!(output[..], [0.0, 3.0, 0.0, 0.0]);
    }
}
