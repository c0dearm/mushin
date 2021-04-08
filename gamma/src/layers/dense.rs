use crate::activations::Activation;
use core::mem::MaybeUninit;
use rand::{distributions::Distribution, Rng, RngCore};

#[derive(Debug)]
pub struct Dense<const I: usize, const O: usize>
where
    [(); I + 1]: ,
{
    weights: [[f32; I + 1]; O], // One extra column for bias weights
    activation: Activation,
}

impl<const I: usize, const O: usize> Dense<I, O>
where
    [(); I + 1]: ,
{
    pub fn new(weights: [[f32; I + 1]; O], activation: Activation) -> Self {
        Dense {
            weights,
            activation,
        }
    }

    pub fn random<R: RngCore, D: Distribution<f32>>(
        rng: &mut R,
        dist: &D,
        activation: Activation,
    ) -> Self {
        let mut weights = [[0.0; I + 1]; O];

        weights
            .iter_mut()
            .flatten()
            .zip(rng.sample_iter(dist))
            .for_each(|(w, r)| *w = r);

        Dense::new(weights, activation)
    }

    pub fn forward(&self, input: [f32; I]) -> [f32; O] {
        let input = input.iter().chain([1.0].iter()); // Add 1.0 to input for bias weights

        let mut output = MaybeUninit::uninit_array();

        for (k, o) in output.iter_mut().enumerate() {
            *o = MaybeUninit::new((self.activation)(
                input
                    .clone()
                    .zip(self.weights[k].iter())
                    .fold(0.0, |acc, (x, y)| acc + x * y),
            ));
        }

        unsafe { MaybeUninit::array_assume_init(output) }
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
        let layer = Dense::<2, 2>::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], relu);
        assert_eq!(layer.weights, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    }

    #[test]
    fn dense_random() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let between = Uniform::from(-1.0..=1.0);
        let layer = Dense::<2, 2>::random(&mut rng, &between, relu);
        assert_eq!(
            layer.weights,
            [
                [-0.6255188, 0.67383957, 0.8181262],
                [0.26284897, 0.5238807, -0.53516835]
            ]
        );
    }

    #[test]
    fn dense_forward() {
        let layer = Dense::<2, 4>::new(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-2.0, -2.0, -2.0],
                [-2.0, -2.0, -2.0],
            ],
            relu,
        );
        let output = layer.forward([1.0, 1.0]);
        assert!((output[0] - 3.0).abs() < f32::EPSILON);
        assert!((output[1] - 3.0).abs() < f32::EPSILON);
        assert!((output[2] - 0.0).abs() < f32::EPSILON);
        assert!((output[3] - 0.0).abs() < f32::EPSILON);
    }
}
