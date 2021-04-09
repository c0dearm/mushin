pub trait Activation<const O: usize> {
    fn activation(x: &mut [f32; O]);
}

#[derive(Debug)]
pub struct Nope;
#[derive(Debug)]
pub struct ReLu;
#[derive(Debug)]
pub struct Sigmoid;
#[derive(Debug)]
pub struct Softmax;

impl<const O: usize> Activation<O> for Nope {
    fn activation(_x: &mut [f32; O]) {}
}

impl<const O: usize> Activation<O> for ReLu {
    fn activation(x: &mut [f32; O]) {
        x.iter_mut().for_each(|x| *x = x.max(0.0));
    }
}

impl<const O: usize> Activation<O> for Sigmoid {
    fn activation(x: &mut [f32; O]) {
        x.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp()));
    }
}

// TODO: I hate iterating twice, I wonder if it can be improved
impl<const O: usize> Activation<O> for Softmax {
    fn activation(x: &mut [f32; O]) {
        let sum = x.iter_mut().fold(0.0, |acc, x| {
            *x = x.exp();
            acc + *x
        });
        x.iter_mut().for_each(|x| *x /= sum);
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::{Nope, ReLu, Sigmoid, Softmax};

    #[test]
    fn nope_activation() {
        let mut x = [1.0];
        Nope::activation(&mut x);
        approx::assert_relative_eq!(x[..], [1.0]);
    }

    #[test]
    fn relu_activation() {
        let mut x = [-1.0, 1.0];
        ReLu::activation(&mut x);
        approx::assert_relative_eq!(x[..], [0.0, 1.0]);
    }

    #[test]
    fn sigmoid_activation() {
        let mut x = [-1.0, 1.0];
        Sigmoid::activation(&mut x);
        approx::assert_relative_eq!(x[..], [0.26894143, 0.7310586]);
    }

    #[test]
    fn softmax_activation() {
        let mut x = [0.5, 0.9, 0.1];
        Softmax::activation(&mut x);
        approx::assert_relative_eq!(x[..], [0.31624106, 0.47177622, 0.21198273]);
    }
}
