use crate::tensor::{
    traits::{Data, Pair, Tensed},
    Tensor,
};
use arrayfire::Array;

/// Changes the shape of the tensor to the given dimensions
#[inline]
pub fn reshape<const B: u64, const C: u64, const H: u64, const W: u64, X: Tensed>(
    x: &X,
) -> Tensor<B, C, H, W, X::Data> {
    x.push_unary(
        arrayfire::moddims(&x.data(), arrayfire::dim4!(H, W, C, B)),
        |df: &Array<f32>, _: &[Array<f32>]| {
            arrayfire::moddims(
                df,
                arrayfire::dim4!(X::HEIGHT, X::WIDTH, X::CHANNELS, X::BATCH),
            )
        },
        &[],
    )
}

/// Sine operation
#[inline]
pub fn sin<X: Tensed>(
    x: &X,
) -> Tensor<{ X::BATCH }, { X::CHANNELS }, { X::HEIGHT }, { X::WIDTH }, X::Data> {
    x.push_unary(
        arrayfire::sin(&x.data()),
        |df: &Array<f32>, args: &[Array<f32>]| df * arrayfire::cos(&args[0]),
        &[x.data()],
    )
}

/// Cosine operation
#[inline]
pub fn cos<X: Tensed>(
    x: &X,
) -> Tensor<{ X::BATCH }, { X::CHANNELS }, { X::HEIGHT }, { X::WIDTH }, X::Data> {
    x.push_unary(
        arrayfire::cos(&x.data()),
        |df: &Array<f32>, args: &[Array<f32>]| df * -arrayfire::sin(&args[0]),
        &[x.data()],
    )
}

/// Element-wise addition
#[inline]
pub fn add<X: Tensed, Y: Data>(
    x: &X,
    y: &Tensor<{ X::BATCH | 1 }, { X::CHANNELS }, { X::HEIGHT }, { X::WIDTH }, Y>,
) -> Tensor<{ X::BATCH }, { X::CHANNELS }, { X::HEIGHT }, { X::WIDTH }, <X::Data as Pair<Y>>::Output>
where
    X::Data: Pair<Y>,
{
    x.push_binary(
        y,
        arrayfire::add(&x.data(), &y.data(), true),
        |df: &Array<f32>, _: &[Array<f32>]| (df.clone(), df.clone()),
        &[],
    )
}

/// Element-wise substraction
#[inline]
pub fn sub<X: Tensed, Y: Data>(
    x: &X,
    y: &Tensor<{ X::BATCH | 1 }, { X::CHANNELS }, { X::HEIGHT }, { X::WIDTH }, Y>,
) -> Tensor<{ X::BATCH }, { X::CHANNELS }, { X::HEIGHT }, { X::WIDTH }, <X::Data as Pair<Y>>::Output>
where
    X::Data: Pair<Y>,
{
    x.push_binary(
        y,
        arrayfire::sub(&x.data(), &y.data(), true),
        |df: &Array<f32>, _: &[Array<f32>]| (df.clone(), -df.clone()),
        &[],
    )
}

/// Element-wise multiplication
#[inline]
pub fn mul<X: Tensed, Y: Data>(
    x: &X,
    y: &Tensor<{ X::BATCH | 1 }, { X::CHANNELS }, { X::HEIGHT }, { X::WIDTH }, Y>,
) -> Tensor<{ X::BATCH }, { X::CHANNELS }, { X::HEIGHT }, { X::WIDTH }, <X::Data as Pair<Y>>::Output>
where
    X::Data: Pair<Y>,
{
    x.push_binary(
        y,
        arrayfire::mul(&x.data(), &y.data(), true),
        |df: &Array<f32>, args: &[Array<f32>]| (df * &args[1], df * &args[0]),
        &[x.data(), y.data()],
    )
}

/// Element-wise division
#[inline]
pub fn div<X: Tensed, Y: Data>(
    x: &X,
    y: &Tensor<{ X::BATCH | 1 }, { X::CHANNELS }, { X::HEIGHT }, { X::WIDTH }, Y>,
) -> Tensor<{ X::BATCH }, { X::CHANNELS }, { X::HEIGHT }, { X::WIDTH }, <X::Data as Pair<Y>>::Output>
where
    X::Data: Pair<Y>,
{
    x.push_binary(
        y,
        arrayfire::div(&x.data(), &y.data(), false),
        |df: &Array<f32>, args: &[Array<f32>]| {
            let (a, b) = (&args[0], &args[1]);
            (df / b, -(df * a / b / b))
        },
        &[x.data(), y.data()],
    )
}

/// Common matrix multiplication
#[inline]
pub fn mm<X, Y>(
    x: &X,
    y: &Y,
) -> Tensor<
    { X::BATCH },
    { X::CHANNELS },
    { X::HEIGHT },
    { Y::WIDTH },
    <X::Data as Pair<Y::Data>>::Output,
>
where
    X: Tensed,
    Y: Tensed<BATCH = 1, CHANNELS = 1, HEIGHT = { X::WIDTH }>,
    X::Data: Pair<Y::Data>,
{
    let reverse = |df: &Array<f32>, args: &[Array<f32>]| {
        (
            arrayfire::matmul(
                df,
                &args[1],
                arrayfire::MatProp::NONE,
                arrayfire::MatProp::TRANS,
            ),
            arrayfire::matmul(
                &args[0],
                df,
                arrayfire::MatProp::TRANS,
                arrayfire::MatProp::NONE,
            ),
        )
    };

    x.push_binary(
        y,
        arrayfire::matmul(
            &x.data(),
            &y.data(),
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE,
        ),
        reverse,
        &[x.data(), y.data()],
    )
}

#[cfg(test)]
mod tests {
    use super::{add, cos, div, mm, mul, reshape, sin, sub, Tensed};
    use crate as mu;
    use crate::tests::equal_data;
    use arrayfire::{constant, dim4, Array};

    // All result comparisons are taken from performing the exact same operations on Tensorflow

    #[test]
    fn reshape_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let z = reshape::<1, 1, 1, 6, _>(&x);
        assert!(equal_data(
            z.data(),
            Array::new(&[3.0, 0.0, 0.0, 0.0, 3.0, 0.0], dim4!(1, 6, 1, 1))
        ));

        z.backward();
        assert!(equal_data(
            x.grad().data(),
            Array::new(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dim4!(3, 2, 1, 1))
        ));
    }

    #[test]
    fn sin_forward_backward() {
        let x = mu::eye::<1, 1, 2, 3>(0.5);
        let z = sin(&x);
        assert!(equal_data(
            z.data(),
            Array::new(
                &[0.479425538604203, 0.0, 0.0, 0.479425538604203, 0.0, 0.0],
                dim4!(2, 3, 1, 1),
            ),
        ));

        z.backward();
        assert!(equal_data(
            x.grad().data(),
            Array::new(
                &[0.8775825618903728, 1.0, 1.0, 0.8775825618903728, 1.0, 1.0],
                dim4!(2, 3, 1, 1),
            ),
        ))
    }

    #[test]
    fn cos_forward_backward() {
        let x = mu::eye::<1, 1, 2, 3>(0.5);
        let z = cos(&x);
        assert!(equal_data(
            z.data(),
            Array::new(
                &[0.8775825618903728, 1.0, 1.0, 0.8775825618903728, 1.0, 1.0],
                dim4!(2, 3, 1, 1),
            ),
        ));

        z.backward();
        assert!(equal_data(
            x.grad().data(),
            Array::new(
                &[-0.479425538604203, 0.0, 0.0, -0.479425538604203, 0.0, 0.0],
                dim4!(2, 3, 1, 1),
            ),
        ));
    }

    #[test]
    fn add_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let y = mu::fill::<1, 1, 3, 2>(2.0);
        let z = add(&x, &y);
        assert!(equal_data(
            z.data(),
            Array::new(&[5.0, 2.0, 2.0, 2.0, 5.0, 2.0], dim4!(3, 2, 1, 1))
        ));

        z.backward();
        assert!(equal_data(x.grad().data(), constant!(1.0; 3,2,1,1)));
        assert!(equal_data(y.grad().data(), constant!(1.0; 3,2,1,1)));
    }

    #[test]
    fn sub_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let y = mu::fill::<1, 1, 3, 2>(2.0);
        let z = sub(&x, &y);
        assert!(equal_data(
            z.data(),
            Array::new(&[1.0, -2.0, -2.0, -2.0, 1.0, -2.0], dim4!(3, 2, 1, 1))
        ));

        z.backward();
        assert!(equal_data(x.grad().data(), constant!(1.0; 3,2,1,1)));
        assert!(equal_data(y.grad().data(), constant!(-1.0; 3,2,1,1)));
    }

    #[test]
    fn mul_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let y = mu::fill::<1, 1, 3, 2>(2.0);
        let z = mul(&x, &y);
        assert!(equal_data(
            z.data(),
            Array::new(&[6.0, 0.0, 0.0, 0.0, 6.0, 0.0], dim4!(3, 2, 1, 1))
        ));

        z.backward();
        assert!(equal_data(x.grad().data(), constant!(2.0; 3,2,1,1)));
        assert!(equal_data(
            y.grad().data(),
            arrayfire::identity::<f32>(dim4!(3, 2, 1, 1)) * 3.0f32
        ));
    }

    #[test]
    fn div_forward_backward() {
        let x = mu::fill::<1, 1, 3, 2>(2.0);
        let y = mu::fill::<1, 1, 3, 2>(4.0);
        let z = div(&x, &y);
        assert!(equal_data(z.data(), constant!(0.5; 3, 2, 1, 1)));

        z.backward();
        assert!(equal_data(x.grad().data(), constant!(0.25; 3,2,1,1)));
        assert!(equal_data(y.grad().data(), constant!(-0.125; 3,2,1,1)));
    }

    #[test]
    fn mm_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let y = mu::eye::<1, 1, 2, 4>(2.0);
        let z = mm(&x, &y);
        assert!(equal_data(
            z.data(),
            Array::new(
                &[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dim4!(3, 4, 1, 1),
            ),
        ));

        z.backward();
        assert!(equal_data(x.grad().data(), constant!(2.0; 3,2,1,1)));
        assert!(equal_data(y.grad().data(), constant!(3.0; 2,4,1,1)));
    }
}
