use arrayfire::{Array, MatProp};

use crate::context::function::Function;
use crate::tensor::Tensor;

/// Computes the element-wise sinus function
#[must_use]
#[inline]
pub fn sin<'ctx, const B: u64, const L: u64, const R: u64, const C: u64>(
    x: &'ctx Tensor<'ctx, B, L, R, C>,
) -> Tensor<'ctx, B, L, R, C> {
    let backward = |df: &Array<f32>, a: &Array<f32>| df * arrayfire::cos(a);

    Tensor::new(
        arrayfire::sin(x.into()),
        Function::unary(x, backward),
        x.context(),
    )
}

/// Performs the element-wise addition
#[must_use]
#[inline]
pub fn add<'ctx, const B: u64, const L: u64, const R: u64, const C: u64>(
    x: &'ctx Tensor<'ctx, B, L, R, C>,
    y: &Tensor<B, L, R, C>,
) -> Tensor<'ctx, B, L, R, C> {
    let backward = |df: &Array<f32>, _: &Array<f32>, _: &Array<f32>| (df.clone(), df.clone());

    Tensor::new(
        arrayfire::add(&Array::from(x), &Array::from(y), false),
        Function::binary(x, y, backward),
        x.context(),
    )
}

/// Performs the element-wise substraction
#[must_use]
#[inline]
pub fn sub<'ctx, const B: u64, const L: u64, const R: u64, const C: u64>(
    x: &'ctx Tensor<'ctx, B, L, R, C>,
    y: &Tensor<B, L, R, C>,
) -> Tensor<'ctx, B, L, R, C> {
    let backward = |df: &Array<f32>, _: &Array<f32>, _: &Array<f32>| (df.clone(), -df.clone());

    Tensor::new(
        arrayfire::sub(&Array::from(x), &Array::from(y), false),
        Function::binary(x, y, backward),
        x.context(),
    )
}

/// Performs the common matrix multiplication
#[must_use]
#[inline]
pub fn matmul<'ctx, const B: u64, const L: u64, const R: u64, const C: u64, const YC: u64>(
    x: &'ctx Tensor<'ctx, B, L, R, C>,
    y: &Tensor<B, L, C, YC>,
) -> Tensor<'ctx, B, L, R, YC> {
    let backward = |df: &Array<f32>, a: &Array<f32>, b: &Array<f32>| {
        (
            arrayfire::matmul(df, b, MatProp::NONE, MatProp::TRANS),
            arrayfire::matmul(a, df, MatProp::TRANS, MatProp::NONE),
        )
    };

    Tensor::new(
        arrayfire::matmul(
            &Array::from(x),
            &Array::from(y),
            MatProp::NONE,
            MatProp::NONE,
        ),
        Function::binary(x, y, backward),
        x.context(),
    )
}

/// Performs the Hadamard product (element-wise multiplication of two tensors)
#[must_use]
#[inline]
pub fn multiply<'ctx, const B: u64, const L: u64, const R: u64, const C: u64>(
    x: &'ctx Tensor<'ctx, B, L, R, C>,
    y: &Tensor<B, L, R, C>,
) -> Tensor<'ctx, B, L, R, C> {
    let backward = |df: &Array<f32>, a: &Array<f32>, b: &Array<f32>| (df * b, df * a);

    Tensor::new(
        arrayfire::mul(&Array::from(x), &Array::from(y), false),
        Function::binary(x, y, backward),
        x.context(),
    )
}

/// Computes the element-wise power of two tensors
#[must_use]
#[inline]
pub fn pow<'ctx, const B: u64, const L: u64, const R: u64, const C: u64>(
    x: &'ctx Tensor<'ctx, B, L, R, C>,
    y: &Tensor<B, L, R, C>,
) -> Tensor<'ctx, B, L, R, C> {
    let backward = |df: &Array<f32>, a: &Array<f32>, b: &Array<f32>| {
        (
            df * b * arrayfire::pow(a, &(b - 1.0f32), false),
            df * arrayfire::pow(a, b, false) * arrayfire::log(a),
        )
    };

    Tensor::new(
        arrayfire::pow(&Array::from(x), &Array::from(y), false),
        Function::binary(x, y, backward),
        x.context(),
    )
}

/// Computes the sum of all the elements in the tensor
#[must_use]
#[inline]
pub fn sum<'ctx, const B: u64, const L: u64, const R: u64, const C: u64>(
    x: &'ctx Tensor<'ctx, B, L, R, C>,
) -> Tensor<'ctx, 1, 1, 1, 1> {
    let backward = |df: &Array<f32>, _: &Array<f32>| df.clone();

    let (value, _) = arrayfire::sum_all(x.into());

    Tensor::new(
        arrayfire::constant!(value; 1,1,1,1),
        Function::unary(x, backward),
        x.context(),
    )
}

/// Computes the element-wise division of two tensors
#[must_use]
#[inline]
pub fn div<'ctx, const B: u64, const L: u64, const R: u64, const C: u64>(
    x: &'ctx Tensor<'ctx, B, L, R, C>,
    y: &Tensor<'ctx, B, L, R, C>,
) -> Tensor<'ctx, B, L, R, C> {
    let backward = |df: &Array<f32>, a: &Array<f32>, b: &Array<f32>| (df / b, -(df * a / b / b));

    Tensor::new(
        arrayfire::div(&Array::from(x), &Array::from(y), false),
        Function::binary(x, y, backward),
        x.context(),
    )
}

#[cfg(test)]
mod tests {
    use arrayfire::{abs, all_true_all, constant, dim4, le, Array};

    use crate::{Class, Context, Gradients, Values};

    use super::*;

    // Helper function to assert that two arryfire Arrays are equal
    fn assert_equal(x: &Array<f32>, y: &Array<f32>) {
        assert!(all_true_all(&le(&abs(&(x - y)), &1e-15, false)).0)
    }

    #[test]
    fn sin_forward_backward() {
        let ctx = Context::new();
        let x = ctx.tensor::<1, 1, 2, 3>(Values::Eye(0.5), Class::Variable);
        let z = sin(&x);
        assert_equal(
            &Array::from(&z),
            &Array::new(
                &[0.479425538604203, 0.0, 0.0, 0.479425538604203, 0.0, 0.0],
                dim4!(2, 3, 1, 1),
            ),
        );

        let grads = Gradients::compute(&z);
        assert_equal(
            &grads.wrt(&x).into(),
            &Array::new(
                &[0.8775825618903728, 1.0, 1.0, 0.8775825618903728, 1.0, 1.0],
                dim4!(2, 3, 1, 1),
            ),
        );
    }

    #[test]
    fn add_forward_backward() {
        let ctx = Context::new();
        let x = ctx.tensor::<1, 1, 3, 2>(Values::Eye(3.0), Class::Variable);
        let y = ctx.tensor::<1, 1, 3, 2>(Values::Fill(2.0), Class::Variable);
        let z = add(&x, &y);
        assert_equal(
            &Array::from(&z),
            &Array::new(&[5.0, 2.0, 2.0, 2.0, 5.0, 2.0], dim4!(3, 2, 1, 1)),
        );

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x).into(), &constant!(1.0; 3,2,1,1));
        assert_equal(&grads.wrt(&y).into(), &constant!(1.0; 3,2,1,1));
    }

    #[test]
    fn sub_forward_backward() {
        let ctx = Context::new();
        let x = ctx.tensor::<1, 1, 3, 2>(Values::Eye(3.0), Class::Variable);
        let y = ctx.tensor::<1, 1, 3, 2>(Values::Fill(2.0), Class::Variable);
        let z = sub(&x, &y);
        assert_equal(
            &Array::from(&z),
            &Array::new(&[1.0, -2.0, -2.0, -2.0, 1.0, -2.0], dim4!(3, 2, 1, 1)),
        );

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x).into(), &constant!(1.0; 3,2,1,1));
        assert_equal(&grads.wrt(&y).into(), &constant!(-1.0; 3,2,1,1));
    }

    #[test]
    fn matmul_forward_backward() {
        let ctx = Context::new();
        let x = ctx.tensor::<1, 1, 3, 2>(Values::Eye(3.0), Class::Variable);
        let y = ctx.tensor::<1, 1, 2, 4>(Values::Eye(2.0), Class::Variable);
        let z = matmul(&x, &y);
        assert_equal(
            &Array::from(&z),
            &Array::new(
                &[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dim4!(3, 4, 1, 1),
            ),
        );

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x).into(), &constant!(2.0f32; 3, 2, 1, 1));
        assert_equal(&grads.wrt(&y).into(), &constant!(3.0f32; 2, 4, 1, 1));
    }

    #[test]
    fn multiply_forward_backward() {
        let ctx = Context::new();
        let x = ctx.tensor::<1, 1, 2, 3>(Values::Fill(3.0), Class::Variable);
        let y = ctx.tensor::<1, 1, 2, 3>(Values::Fill(2.0), Class::Variable);
        let z = multiply(&x, &y);
        assert_equal(&Array::from(&z), &constant!(6.0f32; 2, 3, 1, 1));

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x).into(), &constant!(2.0f32; 2, 3, 1, 1));
        assert_equal(&grads.wrt(&y).into(), &constant!(3.0f32; 2, 3, 1, 1));
    }

    #[test]
    fn pow_forward_backward() {
        let ctx = Context::new();
        let x = ctx.tensor::<1, 1, 3, 2>(Values::Fill(2.0), Class::Variable);
        let y = ctx.tensor::<1, 1, 3, 2>(Values::Fill(3.0), Class::Variable);
        let z = pow(&x, &y);
        assert_equal(&Array::from(&z), &constant!(8.0; 3,2,1,1));

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x).into(), &constant!(12.0; 3,2,1,1));
        assert_equal(&grads.wrt(&y).into(), &constant!(5.5451775; 3,2,1,1));
    }

    #[test]
    fn sum_forward_backward() {
        let ctx = Context::new();
        let x = ctx.tensor::<1, 1, 3, 2>(Values::Fill(2.0), Class::Variable);
        let z = sum(&x);
        assert_equal(&Array::from(&z), &Array::new(&[12.0], dim4!(1, 1, 1, 1)));

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x).into(), &constant!(1.0; 3,2,1,1));
    }

    #[test]
    fn div_forward_backward() {
        let ctx = Context::new();
        let x = ctx.tensor::<1, 1, 3, 2>(Values::Fill(2.0), Class::Variable);
        let y = ctx.tensor::<1, 1, 3, 2>(Values::Fill(4.0), Class::Variable);
        let z = div(&x, &y);
        assert_equal(
            &Array::from(&z),
            &Array::new(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dim4!(3, 2, 1, 1)),
        );

        let grads = Gradients::compute(&z);
        assert_equal(&grads.wrt(&x).into(), &constant!(0.25; 3,2,1,1));
        assert_equal(&grads.wrt(&y).into(), &constant!(-0.125; 3,2,1,1));
    }
}
