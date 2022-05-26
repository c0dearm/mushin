use arrayfire::{constant, Array, MatProp};

use crate::tensor::{BinaryOp, Pair, Tensor, UnaryOp};

/// Computes the element-wise sinus function
#[must_use]
#[inline]
pub fn sin<const B: u64, const L: u64, const R: u64, const C: u64, X, Y>(x: &X) -> Y
where
    X: Tensor<B, L, R, C> + UnaryOp<Y>,
    Y: Tensor<B, L, R, C>,
{
    let forward = |a: &Array<f32>| arrayfire::sin(a);
    let reverse = |df: &Array<f32>, a: &Array<f32>| df * arrayfire::cos(a);

    x.eval(forward, reverse)
}

/// Computes the sum of all the elements in the tensor
#[must_use]
#[inline]
pub fn sum<const B: u64, const L: u64, const R: u64, const C: u64, X, Y>(x: &X) -> Y
where
    X: Tensor<B, L, R, C> + UnaryOp<Y>,
    Y: Tensor<1, 1, 1, 1>,
{
    let forward = |a: &Array<f32>| constant!(arrayfire::sum_all(a).0; 1,1,1,1);
    let reverse = |df: &Array<f32>, _: &Array<f32>| df.clone();

    x.eval(forward, reverse)
}

/// Performs the element-wise addition
#[must_use]
#[inline]
pub fn add<'t, const B: u64, const L: u64, const R: u64, const C: u64, X, Y, Z>(
    x: &'t X,
    y: &'t Y,
) -> Z
where
    X: Tensor<B, L, R, C>,
    Y: Tensor<B, L, R, C>,
    Z: Tensor<B, L, R, C>,
    Pair<'t, X, Y>: BinaryOp<Z>,
{
    let forward = |a: &Array<f32>, b: &Array<f32>| arrayfire::add(a, b, false);
    let reverse = |df: &Array<f32>, _: &Array<f32>, _: &Array<f32>| (df.clone(), df.clone());

    Pair(x, y).eval(forward, reverse)
}

/// Performs the element-wise substraction
#[must_use]
#[inline]
pub fn sub<'t, const B: u64, const L: u64, const R: u64, const C: u64, X, Y, Z>(
    x: &'t X,
    y: &'t Y,
) -> Z
where
    X: Tensor<B, L, R, C>,
    Y: Tensor<B, L, R, C>,
    Z: Tensor<B, L, R, C>,
    Pair<'t, X, Y>: BinaryOp<Z>,
{
    let forward = |a: &Array<f32>, b: &Array<f32>| arrayfire::sub(a, b, false);
    let reverse = |df: &Array<f32>, _: &Array<f32>, _: &Array<f32>| (df.clone(), -df.clone());

    Pair(x, y).eval(forward, reverse)
}

/// Performs the Hadamard product (element-wise multiplication of two tensors)
#[must_use]
#[inline]
pub fn mul<'t, const B: u64, const L: u64, const R: u64, const C: u64, X, Y, Z>(
    x: &'t X,
    y: &'t Y,
) -> Z
where
    X: Tensor<B, L, R, C>,
    Y: Tensor<B, L, R, C>,
    Z: Tensor<B, L, R, C>,
    Pair<'t, X, Y>: BinaryOp<Z>,
{
    let forward = |a: &Array<f32>, b: &Array<f32>| arrayfire::mul(a, b, false);
    let reverse = |df: &Array<f32>, a: &Array<f32>, b: &Array<f32>| (df * b, df * a);

    Pair(x, y).eval(forward, reverse)
}

/// Computes the element-wise division of two tensors
#[must_use]
#[inline]
pub fn div<'t, const B: u64, const L: u64, const R: u64, const C: u64, X, Y, Z>(
    x: &'t X,
    y: &'t Y,
) -> Z
where
    X: Tensor<B, L, R, C>,
    Y: Tensor<B, L, R, C>,
    Z: Tensor<B, L, R, C>,
    Pair<'t, X, Y>: BinaryOp<Z>,
{
    let forward = |a: &Array<f32>, b: &Array<f32>| arrayfire::div(a, b, false);
    let reverse = |df: &Array<f32>, a: &Array<f32>, b: &Array<f32>| (df / b, -(df * a / b / b));

    Pair(x, y).eval(forward, reverse)
}

/// Computes the element-wise power of two tensors
#[must_use]
#[inline]
pub fn pow<'t, const B: u64, const L: u64, const R: u64, const C: u64, X, Y, Z>(
    x: &'t X,
    y: &'t Y,
) -> Z
where
    X: Tensor<B, L, R, C>,
    Y: Tensor<B, L, R, C>,
    Z: Tensor<B, L, R, C>,
    Pair<'t, X, Y>: BinaryOp<Z>,
{
    let forward = |a: &Array<f32>, b: &Array<f32>| arrayfire::pow(a, b, false);
    let reverse = |df: &Array<f32>, a: &Array<f32>, b: &Array<f32>| {
        (
            df * b * arrayfire::pow(a, &(b - 1.0f32), false),
            df * arrayfire::pow(a, b, false) * arrayfire::log(a),
        )
    };

    Pair(x, y).eval(forward, reverse)
}

/// Performs the common matrix multiplication
#[must_use]
#[inline]
pub fn matmul<'t, const B: u64, const L: u64, const R: u64, const C: u64, const YC: u64, X, Y, Z>(
    x: &'t X,
    y: &'t Y,
) -> Z
where
    X: Tensor<B, L, R, C>,
    Y: Tensor<B, L, C, YC>,
    Z: Tensor<B, L, R, YC>,
    Pair<'t, X, Y>: BinaryOp<Z>,
{
    let forward =
        |a: &Array<f32>, b: &Array<f32>| arrayfire::matmul(a, b, MatProp::NONE, MatProp::NONE);
    let reverse = |df: &Array<f32>, a: &Array<f32>, b: &Array<f32>| {
        (
            arrayfire::matmul(df, b, MatProp::NONE, MatProp::TRANS),
            arrayfire::matmul(a, df, MatProp::TRANS, MatProp::NONE),
        )
    };

    Pair(x, y).eval(forward, reverse)
}

#[cfg(test)]
mod tests {
    use arrayfire::{abs, all_true_all, constant, dim4, le, Array};

    use crate::{Context, Gradients, Values};

    use super::*;

    // Helper function to assert that two arryfire Arrays are equal
    fn assert_equal<
        const B: u64,
        const L: u64,
        const R: u64,
        const C: u64,
        T: Tensor<B, L, R, C>,
    >(
        x: &T,
        y: Array<f32>,
    ) {
        assert!(all_true_all(&le(&abs(&(x.value() - y)), &1e-15, false)).0)
    }

    #[test]
    fn sin_forward_backward() {
        let ctx = Context::new();
        let x = ctx.variable::<1, 1, 2, 3>(Values::Eye(0.5), None);
        let z = sin(&x);
        assert_equal(
            &z,
            Array::new(
                &[0.479425538604203, 0.0, 0.0, 0.479425538604203, 0.0, 0.0],
                dim4!(2, 3, 1, 1),
            ),
        );

        let grads = Gradients::compute(&z);
        assert_equal(
            &grads.wrt(&x),
            Array::new(
                &[0.8775825618903728, 1.0, 1.0, 0.8775825618903728, 1.0, 1.0],
                dim4!(2, 3, 1, 1),
            ),
        );
    }

    #[test]
    fn sum_forward_backward() {
        let ctx = Context::new();
        let x = ctx.variable::<1, 1, 3, 2>(Values::Fill(2.0), None);
        let z = sum(&x);
        assert_equal(&z, Array::new(&[12.0], dim4!(1, 1, 1, 1)));

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x), constant!(1.0; 3,2,1,1));
    }

    #[test]
    fn add_forward_backward() {
        let ctx = Context::new();
        let x = ctx.variable::<1, 1, 3, 2>(Values::Eye(3.0), None);
        let y = ctx.variable::<1, 1, 3, 2>(Values::Fill(2.0), None);
        let z = add(&x, &y);
        assert_equal(
            &z,
            Array::new(&[5.0, 2.0, 2.0, 2.0, 5.0, 2.0], dim4!(3, 2, 1, 1)),
        );

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x), constant!(1.0; 3,2,1,1));
        assert_equal(&grads.wrt(&y), constant!(1.0; 3,2,1,1));
    }

    #[test]
    fn sub_forward_backward() {
        let ctx = Context::new();
        let x = ctx.variable::<1, 1, 3, 2>(Values::Eye(3.0), None);
        let y = ctx.variable::<1, 1, 3, 2>(Values::Fill(2.0), None);
        let z = sub(&x, &y);
        assert_equal(
            &z,
            Array::new(&[1.0, -2.0, -2.0, -2.0, 1.0, -2.0], dim4!(3, 2, 1, 1)),
        );

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x), constant!(1.0; 3,2,1,1));
        assert_equal(&grads.wrt(&y), constant!(-1.0; 3,2,1,1));
    }

    #[test]
    fn mul_forward_backward() {
        let ctx = Context::new();
        let x = ctx.variable::<1, 1, 2, 3>(Values::Fill(3.0), None);
        let y = ctx.variable::<1, 1, 2, 3>(Values::Fill(2.0), None);
        let z = mul(&x, &y);
        assert_equal(&z, constant!(6.0f32; 2, 3, 1, 1));

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x), constant!(2.0f32; 2, 3, 1, 1));
        assert_equal(&grads.wrt(&y), constant!(3.0f32; 2, 3, 1, 1));
    }

    #[test]
    fn div_forward_backward() {
        let ctx = Context::new();
        let x = ctx.variable::<1, 1, 3, 2>(Values::Fill(2.0), None);
        let y = ctx.variable::<1, 1, 3, 2>(Values::Fill(4.0), None);
        let z = div(&x, &y);
        assert_equal(
            &z,
            Array::new(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dim4!(3, 2, 1, 1)),
        );

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x), constant!(0.25; 3,2,1,1));
        assert_equal(&grads.wrt(&y), constant!(-0.125; 3,2,1,1));
    }

    #[test]
    fn pow_forward_backward() {
        let ctx = Context::new();
        let x = ctx.variable::<1, 1, 3, 2>(Values::Fill(2.0), None);
        let y = ctx.variable::<1, 1, 3, 2>(Values::Fill(3.0), None);
        let z = pow(&x, &y);
        assert_equal(&z, constant!(8.0; 3,2,1,1));

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x), constant!(12.0; 3,2,1,1));
        assert_equal(&grads.wrt(&y), constant!(5.5451775; 3,2,1,1));
    }

    #[test]
    fn matmul_forward_backward() {
        let ctx = Context::new();
        let x = ctx.variable::<1, 1, 3, 2>(Values::Eye(3.0), None);
        let y = ctx.variable::<1, 1, 2, 4>(Values::Eye(2.0), None);
        let z = matmul(&x, &y);
        assert_equal(
            &z,
            Array::new(
                &[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dim4!(3, 4, 1, 1),
            ),
        );

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x), constant!(2.0f32; 3, 2, 1, 1));
        assert_equal(&grads.wrt(&y), constant!(3.0f32; 2, 4, 1, 1));
    }
}
