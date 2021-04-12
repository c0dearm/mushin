pub trait Activation {
    fn activation(x: f32) -> f32;
}

#[derive(Debug)]
pub struct Nope;
#[derive(Debug)]
pub struct ReLu;
#[derive(Debug)]
pub struct Sigmoid;
#[derive(Debug)]
pub struct Softmax;

impl Activation for Nope {
    fn activation(x: f32) -> f32 {
        x
    }
}

impl Activation for ReLu {
    fn activation(x: f32) -> f32 {
        x.max(0.0)
    }
}

impl Activation for Sigmoid {
    fn activation(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::{Nope, ReLu, Sigmoid, Softmax};

    #[test]
    fn nope_activation() {
        approx::assert_relative_eq!(Nope::activation(1.0), 1.0);
    }

    #[test]
    fn relu_activation() {
        approx::assert_relative_eq!(ReLu::activation(-1.0), 0.0);
        approx::assert_relative_eq!(ReLu::activation(1.0), 1.0);
    }

    #[test]
    fn sigmoid_activation() {
        approx::assert_relative_eq!(Sigmoid::activation(-1.0), 0.26894143);
        approx::assert_relative_eq!(Sigmoid::activation(1.0), 0.7310586);
    }
}
