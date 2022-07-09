use crate::graph::node::Node;
use std::rc::Rc;

/// Stochastic Gradient Descent
pub struct SGD {
    lr: f32,
    params: Vec<Rc<Node>>,
}

impl SGD {
    #[inline]
    pub fn new<'n, P>(params: &'n P, lr: f32) -> Self
    where
        &'n P: IntoIterator<Item = &'n Rc<Node>>,
    {
        Self {
            lr,
            params: params
                .into_iter()
                .filter_map(|n| {
                    if n.is_declaration() {
                        Some(n.clone())
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    #[inline]
    pub fn step(&self) {
        for node in &self.params {
            let step = arrayfire::sub(
                &node.data().clone(),
                &(self.lr * &node.grad().clone()),
                true,
            );
            *node.data_mut() = step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SGD;
    use crate as mu;
    use crate::tensor::traits::Tensed;
    use crate::tests::equal_data;

    #[test]
    fn sgd_step() {
        let x = mu::fill::<1, 1, 1, 1>(1.0);
        let optim = SGD::new(&[x.inner().node()], 0.1);

        x.backward();
        optim.step();
        assert!(equal_data(x.data(), arrayfire::constant!(0.9; 1,1,1,1)));
    }
}
