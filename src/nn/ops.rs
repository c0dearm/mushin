use crate::{
    ops::reshape,
    tensor::{
        traits::{Data, Tensed},
        Tensor,
    },
};
use arrayfire::{dim4, view, Array, Seq};

// Given an input tensor, returns a tensor that keeps the same batch size, but with the rest
// of the dimensions flattened to a vector.
#[inline]
pub fn flatten<const B: u64, const C: u64, const H: u64, const W: u64, D: Data>(
    x: &Tensor<B, C, H, W, D>,
) -> Tensor<B, 1, 1, { C * H * W }, D> {
    x.push_unary(
        arrayfire::moddims(&x.data(), arrayfire::dim4!(1, { C * H * W }, 1, B)),
        |df: &Array<f32>, _: &[Array<f32>]| arrayfire::moddims(df, arrayfire::dim4!(H, W, C, B)),
        &[],
    )
}

pub struct MaxPool2D<const H: u64, const W: u64, const S: u64>;

impl<const H: u64, const W: u64, const S: u64> MaxPool2D<H, W, S> {
    // Performs the 2-dimensional max pooling operation on a given tensor.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    pub fn forward<const B: u64, const C: u64, const XH: u64, const XW: u64, D: Data>(
        x: &Tensor<B, C, XH, XW, D>,
    ) -> Tensor<B, C, { (XH - H) / S + 1 }, { (XW - W) / S + 1 }, D> {
        let input = x.data();
        let mut values = vec![0.0; (B * C * ((XH - H) / S + 1) * ((XW - H / S) + 1)) as usize];
        let mut indices = vec![0.0; (B * C * XH * XW) as usize];
        let mut count = 0;

        for b in 0..B {
            for c in 0..C {
                for w in (0..XW).step_by(S as usize) {
                    for h in (0..XH).step_by(S as usize) {
                        let batch = Seq::new(b as i32, b as i32, 1);
                        let channel = Seq::new(c as i32, c as i32, 1);
                        let rows = Seq::new(h as i32, (h + H - 1) as i32, 1);
                        let cols = Seq::new(w as i32, (w + W - 1) as i32, 1);
                        let (v, _, i) =
                            arrayfire::imax_all(&view!(input[rows, cols, channel, batch]));

                        let index = i as usize
                            + b as usize * (C * XH * XW) as usize
                            + c as usize * (XH * XW) as usize
                            + w as usize * XH as usize
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
            Array::new(&values[..], dim4!((XH - H) / S + 1, (XW - W) / S + 1, C, B)),
            reverse,
            &[Array::new(&indices[..], dim4!(XH, XW, C, B))],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{flatten, MaxPool2D, Tensed};
    use crate as mu;
    use crate::tests::equal_data;
    use arrayfire::Array;

    #[test]
    fn flatten_forward_backward() {
        let x = mu::fill::<2, 2, 3, 4>(2.0);
        let z = flatten(&x);
        assert!(equal_data(z.data(), arrayfire::constant!(2.0; 1,24,1,2)));

        z.backward();
        assert!(equal_data(
            x.grad().data(),
            arrayfire::constant!(1.0; 3,4,2,2)
        ));
    }

    #[test]
    fn maxpool2d_forward_backward() {
        let x = mu::custom::<1, 1, 4, 4>(&[
            10.0, 4.0, 18.0, 3.0, 12.0, 11.0, 13.0, 15.0, 8.0, 5.0, 7.0, 2.0, 7.0, 9.0, 7.0, 2.0,
        ]);
        let z = MaxPool2D::<2, 2, 2>::forward(&x);
        assert!(equal_data(
            z.data(),
            Array::new(&[12.0, 18.0, 9.0, 7.0], arrayfire::dim4!(2, 2, 1, 1))
        ));

        z.backward();
        assert!(equal_data(
            x.grad().data(),
            Array::new(
                &[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                arrayfire::dim4!(4, 4, 1, 1)
            )
        ));
    }
}
