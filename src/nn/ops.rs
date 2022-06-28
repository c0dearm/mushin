use crate::tensor::{params::SingleParam, Tensor};
use arrayfire::{dim4, view, Array, Seq};

// Given an input tensor, returns a tensor that keeps the same batch size, but with the rest
// of the dimensions flattened to a vector.
#[inline]
pub fn flatten<X>(x: &X) -> X::Out
where
    X: Tensor + SingleParam<{ X::BATCH }, 1, 1, { X::CHANNELS * X::HEIGHT * X::WIDTH }>,
{
    x.reshape()
}

// Performs the 2-dimensional max pooling operation on a given tensor.
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn maxpool2d<const H: u64, const W: u64, const S: u64, X>(x: &X) -> X::Out
where
    X: Tensor
        + SingleParam<{ X::BATCH }, { X::CHANNELS }, { (X::HEIGHT - H) / S }, { (X::WIDTH - W) / S }>,
    X::Out: Tensor,
    [(); (X::BATCH * X::CHANNELS * (X::HEIGHT - H + 2) / S * (X::WIDTH - W + 2) / S) as usize]:,
    [(); ({ X::BATCH } * { X::CHANNELS } * { X::HEIGHT } * { X::WIDTH }) as usize]:,
{
    let input = x.data();
    let values = &mut [0.0; (X::BATCH * X::CHANNELS * (X::HEIGHT - H + 2) / S * (X::WIDTH - W + 2)
        / S) as usize];
    let indices =
        &mut [0.0; ({ X::BATCH } * { X::CHANNELS } * { X::HEIGHT } * { X::WIDTH }) as usize];
    let mut count = 0;

    for b in 0..X::BATCH {
        for c in 0..X::CHANNELS {
            for w in (0..X::WIDTH).step_by(S as usize) {
                for h in (0..X::HEIGHT).step_by(S as usize) {
                    let batch = Seq::new(b as i32, b as i32, 1);
                    let channel = Seq::new(c as i32, c as i32, 1);
                    let rows = Seq::new(h as i32, (h + H - 1) as i32, 1);
                    let cols = Seq::new(w as i32, (w + W - 1) as i32, 1);
                    let (v, _, i) = arrayfire::imax_all(&view!(input[rows, cols, channel, batch]));

                    let index = i as usize
                        + b as usize * (X::CHANNELS * X::HEIGHT * X::WIDTH) as usize
                        + c as usize * (X::HEIGHT * X::WIDTH) as usize
                        + w as usize * X::HEIGHT as usize
                        + h as usize;

                    values[count] = v;
                    indices[index] = 1.0;
                    count += 1;
                }
            }
        }
    }

    let reverse = |_: &Array<f32>, args: &[Array<f32>]| args[0].clone();
    x.push_unary(
        Array::new(
            values,
            dim4!(
                { (X::HEIGHT - H + 2) / S },
                { (X::WIDTH - W + 2) / S },
                { X::CHANNELS },
                { X::BATCH }
            ),
        ),
        reverse,
        &[Array::new(
            indices,
            dim4!({ X::HEIGHT }, { X::WIDTH }, { X::CHANNELS }, { X::BATCH }),
        )],
    )
}

#[cfg(test)]
mod tests {
    use super::{flatten, maxpool2d};
    use crate as mu;
    use crate::tests::equal_arrays;
    use crate::Tensor;
    use arrayfire::Array;

    #[test]
    fn flatten_forward_backward() {
        let x = mu::fill::<2, 2, 3, 4>(2.0);
        let z = flatten(&x);
        assert!(equal_arrays(z.data(), arrayfire::constant!(2.0; 1,24,1,2)));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            arrayfire::constant!(1.0; 3,4,2,2)
        ));
    }

    #[test]
    fn maxpool2d_forward_backward() {
        let x = mu::custom::<1, 1, 4, 4>(&[
            10.0, 4.0, 18.0, 3.0, 12.0, 11.0, 13.0, 15.0, 8.0, 5.0, 7.0, 2.0, 7.0, 9.0, 7.0, 2.0,
        ]);
        let z = maxpool2d::<2, 2, 2, _>(&x);
        assert!(equal_arrays(
            z.data(),
            Array::new(&[12.0, 18.0, 9.0, 7.0], arrayfire::dim4!(2, 2, 1, 1))
        ));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            Array::new(
                &[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                arrayfire::dim4!(4, 4, 1, 1)
            )
        ));
    }
}
