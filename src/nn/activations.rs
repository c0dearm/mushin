use crate::tensor::{params::SingleParam, Tensor};
use arrayfire::{Array, MatProp};

/// Performs the `ReLu` activation function on the given tensor
#[inline]
pub fn relu<X>(x: &X) -> X::Out
where
    X: Tensor + SingleParam<{ X::BATCH }, { X::CHANNELS }, { X::HEIGHT }, { X::WIDTH }>,
{
    let result = arrayfire::maxof(
        &x.data(),
        &arrayfire::constant!(0.0f32; X::HEIGHT,X::WIDTH,X::CHANNELS,X::BATCH),
        false,
    );
    let reverse =
        |df: &Array<f32>, args: &[Array<f32>]| df * arrayfire::gt(&args[0], &0.0f32, false);
    x.push_unary(result, reverse, &[x.data()])
}

/// Performs the `Softmax` activation function on the given row vector
#[inline]
pub fn softmax<X>(x: &X) -> X::Out
where
    X: Tensor<CHANNELS = 1, HEIGHT = 1> + SingleParam<{ X::BATCH }, 1, 1, { X::WIDTH }>,
{
    // This is required for numerical stability
    let shift = arrayfire::sub(&x.data(), &arrayfire::max_all(&x.data()).0, true);
    let exps = arrayfire::exp(&shift);
    let result = arrayfire::div(&exps, &arrayfire::sum_all(&exps).0, false);

    let reverse = |df: &Array<f32>, args: &[Array<f32>]| {
        let softmax = &args[0];
        arrayfire::matmul(
            df,
            &arrayfire::sub(
                &arrayfire::diag_create(&arrayfire::transpose(softmax, false), 0),
                &arrayfire::matmul(softmax, softmax, MatProp::TRANS, MatProp::NONE),
                false,
            ),
            MatProp::NONE,
            MatProp::NONE,
        )
    };

    x.push_unary(result.clone(), reverse, &[result])
}

/// Performs the `log(Softmax)` activation function on the given row vector
#[inline]
pub fn logsoftmax<X>(x: &X) -> X::Out
where
    X: Tensor<CHANNELS = 1, HEIGHT = 1> + SingleParam<{ X::BATCH }, 1, 1, { X::WIDTH }>,
{
    // This is required for numerical stability
    let shift = arrayfire::sub(&x.data(), &arrayfire::max_all(&x.data()).0, true);
    let exps = arrayfire::exp(&shift);
    let softmax = arrayfire::div(&exps, &arrayfire::sum_all(&exps).0, false);
    let result = arrayfire::log(&softmax);

    let reverse = |df: &Array<f32>, args: &[Array<f32>]| {
        let s = &args[0];
        arrayfire::matmul(
            df,
            &arrayfire::sub(
                &arrayfire::identity::<f32>(arrayfire::dim4!(X::WIDTH, X::WIDTH, 1, X::BATCH)),
                &arrayfire::matmul(
                    &arrayfire::constant!(1.0; X::WIDTH, 1, 1, X::BATCH),
                    s,
                    MatProp::NONE,
                    MatProp::NONE,
                ),
                false,
            ),
            MatProp::NONE,
            MatProp::NONE,
        )
    };

    x.push_unary(result, reverse, &[softmax])
}

#[cfg(test)]
mod tests {
    use super::{logsoftmax, relu, softmax};
    use crate as mu;
    use crate::tests::equal_arrays;
    use crate::Tensor;
    use arrayfire::{dim4, Array};

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

    #[test]
    fn softmax_forward_backward() {
        let x = mu::custom::<1, 1, 1, 3>(&[0.3, 0.2, 0.5]);
        let z = softmax(&x);

        assert!(equal_arrays(
            z.data(),
            Array::new(&[0.31987306, 0.28943312, 0.39069384], dim4!(1, 3, 1, 1)),
        ));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            Array::new(&[0.0, 0.0, 0.0], dim4!(1, 3, 1, 1)),
        ));
    }

    #[test]
    fn logsoftmax_forward_backward() {
        let x = mu::custom::<1, 1, 1, 3>(&[0.3, 0.2, 0.5]);
        let z = logsoftmax(&x);

        assert!(equal_arrays(
            z.data(),
            Array::new(&[-1.1398311, -1.239831, -0.939831], dim4!(1, 3, 1, 1)),
        ));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            Array::new(&[0.04038084, 0.13170063, -0.17208147], dim4!(1, 3, 1, 1)),
        ));
    }
}
