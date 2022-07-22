use crate::tensor::{
    constant::Constant,
    traits::{Data, Tensed},
    variable::Variable,
    Tensor,
};
use arrayfire::Array;
use std::marker::PhantomData;

/// A Dropout neural network layer.
/// During training mode (`Dropout<Variable>`) the layer will set values
/// to zero with the given probability. Otherwise it does nothing.
pub struct Dropout<T: Data = Variable>(f32, PhantomData<T>);

impl<T: Data> Dropout<T> {
    #[must_use]
    #[inline]
    pub fn prob(probability: f32) -> Self {
        Self(probability, PhantomData::default())
    }

    #[inline]
    pub fn forward<const B: u64, const C: u64, const H: u64, const W: u64, D: Data>(
        &self,
        x: &Tensor<B, C, H, W, D>,
    ) -> Tensor<B, C, H, W, D> {
        if T::is_constant() {
            x.clone()
        } else {
            let mask =
                arrayfire::gt(&arrayfire::randu!(H, W, C, B), &self.0, false) / (1.0 - self.0);
            let reverse = |df: &Array<f32>, args: &[Array<f32>]| df * &args[0];
            x.push_unary(arrayfire::mul(&x.data(), &mask, false), reverse, &[mask])
        }
    }
}

impl Dropout<Variable> {
    #[must_use]
    #[inline]
    pub fn freeze(self) -> Dropout<Constant> {
        Dropout::prob(self.0)
    }
}

impl Dropout<Constant> {
    #[must_use]
    #[inline]
    pub fn unfreeze(self) -> Dropout<Variable> {
        Dropout::prob(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{Dropout, Variable};
    use crate as mu;
    use crate::tensor::traits::Tensed;
    use crate::tests::equal_data;

    #[test]
    fn dropout_forward_backward() {
        let dropout = Dropout::<Variable>::prob(0.999);
        let x = mu::fill::<1, 1, 1, 1>(2.0);
        let z = dropout.forward(&x);
        assert!(equal_data(z.data(), arrayfire::constant!(0.0; 1,1,1,1)));

        z.backward();
        assert!(equal_data(
            x.grad().data(),
            arrayfire::constant!(0.0; 1,1,1,1)
        ));

        let dropout = dropout.freeze();
        let z = dropout.forward(&x);
        assert!(equal_data(z.data(), arrayfire::constant!(2.0; 1,1,1,1)));

        let dropout = Dropout::<Variable>::prob(0.0);
        let z = dropout.forward(&x);
        assert!(equal_data(z.data(), arrayfire::constant!(2.0; 1,1,1,1)));

        z.reset();
        z.backward();
        assert!(equal_data(
            x.grad().data(),
            arrayfire::constant!(1.0; 1,1,1,1)
        ));
    }
}
