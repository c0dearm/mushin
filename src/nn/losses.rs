use crate::tensor::{
    constant::Constant,
    params::{DoubleParam, SingleParam},
    Tensor,
};

/// Calculates the Mean Squared Error between two tensors
#[inline]
pub fn mse<const B: u64, const L: u64, const R: u64, const C: u64, X, Z, Y, Z0, Z1, Z2>(
    x: &X,
    y: &Y,
) -> Z
where
    Z2: Tensor<1, 1, 1, 1> + DoubleParam<Constant<1, 1, 1, 1>, Z>,
    Z1: Tensor<B, L, R, C> + SingleParam<Z2>,
    Z0: Tensor<B, L, R, C> + DoubleParam<Constant<B, L, R, C>, Z1>,
    X: Tensor<B, L, R, C> + DoubleParam<Y, Z0>,
    Y: Tensor<B, L, R, C>,
    Z: Tensor<1, 1, 1, 1>,
{
    #[allow(clippy::cast_precision_loss)]
    let count = Constant::new(arrayfire::constant!((B*L*R*C) as f32; 1,1,1,1));
    let square = Constant::new(arrayfire::constant!(2.0; R,C,L,B));
    x.sub(y).pow(&square).sum().div(&count)
}

#[cfg(test)]
mod tests {
    use super::mse;
    use crate as mu;
    use crate::tests::equal_arrays;
    use crate::Tensor;

    #[test]
    fn mse_forward_backward() {
        let x = mu::fill::<1, 1, 2, 3>(2.0);
        let y = mu::fill::<1, 1, 2, 3>(0.5);
        let z = mse(&x, &y);
        assert!(equal_arrays(z.data(), arrayfire::constant!(2.25; 1,1,1,1)));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            arrayfire::constant!(0.5; 2,3,1,1)
        ));
        assert!(equal_arrays(
            y.grad().data(),
            arrayfire::constant!(-0.5; 2,3,1,1)
        ));
    }
}
