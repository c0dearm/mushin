use crate::tensor::{
    constant::Constant,
    params::{DoubleParam, SingleParam},
    Tensor,
};

/// A Dropout neural network layer with. Generic `T` indicates if Dropout is in training mode (true) or not (false).
/// During training mode dropout will set some values to zero with the given probability. Otherwise it behaves like
/// the identity function.
pub struct Dropout<const T: bool>(f32);

impl<const T: bool> Dropout<T> {
    #[must_use]
    #[inline]
    pub const fn new(probability: f32) -> Self {
        Self(probability)
    }
}

impl Dropout<true> {
    #[inline]
    pub fn forward<const B: u64, const L: u64, const R: u64, const C: u64, X>(&self, x: &X) -> X
    where
        X: Tensor<B, L, R, C> + DoubleParam<Constant<B, L, R, C>, X>,
    {
        let mask = Constant::new(
            arrayfire::gt(&arrayfire::randu!(B, L, R, C), &self.0, false) / (1.0 - self.0),
        );
        x.mul(&mask)
    }

    #[must_use]
    #[inline]
    pub const fn freeze(self) -> Dropout<false> {
        Dropout::new(self.0)
    }
}

impl Dropout<false> {
    #[allow(clippy::unused_self)]
    #[inline]
    pub fn forward<const B: u64, const L: u64, const R: u64, const C: u64, X>(&self, x: &X) -> X
    where
        X: Tensor<B, L, R, C> + SingleParam<X>,
    {
        x.identity()
    }

    #[must_use]
    #[inline]
    pub const fn unfreeze(self) -> Dropout<true> {
        Dropout::new(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Dropout;
    use crate as mu;
    use crate::tests::equal_arrays;
    use crate::Tensor;

    #[test]
    fn dropout_forward_backward() {
        let dropout = Dropout::<true>::new(0.999);
        let x = mu::fill::<1, 1, 1, 1>(2.0);
        let z = dropout.forward(&x);
        assert!(equal_arrays(z.data(), arrayfire::constant!(0.0; 1,1,1,1)));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            arrayfire::constant!(0.0; 1,1,1,1)
        ));

        let dropout = dropout.freeze();
        let z = dropout.forward(&x);
        assert!(equal_arrays(z.data(), arrayfire::constant!(2.0; 1,1,1,1)));

        let dropout = Dropout::<true>::new(0.0);
        let z = dropout.forward(&x);
        assert!(equal_arrays(z.data(), arrayfire::constant!(2.0; 1,1,1,1)));

        z.reset();
        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            arrayfire::constant!(1.0; 1,1,1,1)
        ));
    }
}
