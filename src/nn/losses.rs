use crate::tensor::{constant::Constant, params::SingleParam, Tensor};
use arrayfire::Array;

/// Calculates the Mean Squared Error between two row vectors
#[inline]
pub fn mse<const B: u64, const C: u64, X, Z>(x: &X, y: &Constant<B, 1, 1, C>) -> Z
where
    X: Tensor<B, 1, 1, C> + SingleParam<Z>,
    Z: Tensor<1, 1, 1, 1>,
{
    let result = arrayfire::div(
        &arrayfire::constant!(arrayfire::sum_all(&arrayfire::pow(
        &arrayfire::sub(&x.data(), &y.data(), false),
        &2.0f32,
        false,
    )).0; 1,1,1,1),
        &C,
        false,
    );

    let reverse = |df: &Array<f32>, args: &[Array<f32>]| {
        df * (2.0f32
            * arrayfire::div(
                &arrayfire::sum_all(&arrayfire::sub(&args[0], &args[1], false)).0,
                &C,
                false,
            ))
    };

    x.push_unary(result, reverse, &[x.data(), y.data()])
}

/// Calculates the Negative Log Likelihood among a set of classes
#[inline]
pub fn nll<const B: u64, const C: u64, X, Z>(x: &X, y: &Constant<B, 1, 1, C>) -> Z
where
    X: Tensor<B, 1, 1, C> + SingleParam<Z>,
    Z: Tensor<1, 1, 1, 1>,
{
    let logits = arrayfire::log(&arrayfire::add(&y.data(), &1e-7f32, false));
    let result = arrayfire::constant!(-arrayfire::sum_all(&arrayfire::mul(
        &x.data(),
        &logits,
        false,
    )).0; 1,1,1,1);

    let reverse = |df: &Array<f32>, args: &[Array<f32>]| -(df * &args[0]);

    x.push_unary(result, reverse, &[logits])
}

#[cfg(test)]
mod tests {
    use super::{mse, nll};
    use crate as mu;
    use crate::tests::equal_arrays;
    use crate::Tensor;
    use arrayfire::Array;

    #[test]
    fn mse_forward_backward() {
        let x = mu::fill::<1, 1, 1, 6>(2.0);
        let y = mu::fill::<1, 1, 1, 6>(0.5).freeze();
        let z = mse(&x, &y);
        assert!(equal_arrays(z.data(), arrayfire::constant!(2.25; 1,1,1,1)));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            arrayfire::constant!(3.0; 1,6,1,1)
        ));
    }

    #[test]
    fn nll_forward_backward() {
        let x = mu::custom::<1, 1, 1, 3>(&[0.5, 0.2, 0.3]);
        let y = mu::custom::<1, 1, 1, 3>(&[1.0, 0.0, 0.0]).freeze();
        let z = nll(&x, &y);
        assert!(equal_arrays(
            z.data(),
            arrayfire::constant!(8.059048; 1,1,1,1)
        ));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            Array::<f32>::new(
                &[1.1920929e-07, 1.6118095e+01, 1.6118095e+01],
                arrayfire::dim4!(1, 3, 1, 1)
            )
        ));
    }
}
