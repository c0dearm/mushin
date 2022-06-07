use crate::tensor::{constant::Constant, params::DoubleParam, Tensor};

/// Performs the `ReLu` activation function on the given tensor
#[inline]
pub fn relu<const B: u64, const L: u64, const R: u64, const C: u64, X>(x: &X) -> X
where
    X: Tensor<B, L, R, C> + DoubleParam<Constant<B, L, R, C>, X>,
{
    x.maximum(&Constant::new(arrayfire::constant!(0.0; R,C,L,B)))
}

#[cfg(test)]
mod tests {
    use super::relu;
    use crate as mu;
    use crate::tests::equal_arrays;
    use crate::Tensor;

    #[test]
    fn relu_forward_backward() {
        let x = mu::custom::<1, 1, 2, 3>(&[1.0, -1.0, -1.0, 1.0, -1.0, -1.0]);
        let z = relu(&x);

        assert!(equal_arrays(
            z.data(),
            arrayfire::identity(arrayfire::dim4!(2, 3, 1, 1))
        ));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            arrayfire::identity(arrayfire::dim4!(2, 3, 1, 1))
        ));
    }
}
