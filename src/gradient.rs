use arrayfire::{constant, Array};

use crate::context::function::{ConstantParameter, DoubleParameter, Function, VariableParameter};
use crate::tensor::{Constant, Values, Variable};

/// Stores the gradients for a given tensor
pub struct Gradients(Vec<Array<f32>>);

impl Gradients {
    /// Given a root variable, computes the derivatives with respect to each of the other variables it depends on
    /// by performing reverse auto-differentiation on its computation graph
    #[must_use]
    #[inline]
    pub fn compute<const B: u64, const N: u64, const R: u64, const C: u64>(
        z: &Variable<B, N, R, C>,
    ) -> Self {
        let mut gradients = vec![constant!(0.0; 1, 1, 1, 1); z.tape().len()];
        gradients[z.index()] = constant!(1.0; R, C, N, B);

        for (i, function) in z.tape().functions().iter().enumerate().rev() {
            match *function {
                Function::Nary => {}
                Function::Unary {
                    param: VariableParameter { ref value, index },
                    reverse,
                } => {
                    let partial = reverse(&gradients[i], value);
                    gradients[index] = &gradients[index] + partial;
                }
                Function::Binary {
                    params:
                        DoubleParameter::VariableConstant(
                            VariableParameter {
                                value: ref value_x,
                                index,
                            },
                            ConstantParameter { value: ref value_y },
                        ),
                    reverse,
                } => {
                    let (partial_x, _) = reverse(&gradients[i], value_x, value_y);
                    gradients[index] = &gradients[index] + partial_x;
                }
                Function::Binary {
                    params:
                        DoubleParameter::ConstantVariable(
                            ConstantParameter { value: ref value_x },
                            VariableParameter {
                                value: ref value_y,
                                index,
                            },
                        ),
                    reverse,
                } => {
                    let (_, partial_y) = reverse(&gradients[i], value_x, value_y);
                    gradients[index] = &gradients[index] + partial_y;
                }
                Function::Binary {
                    params:
                        DoubleParameter::VariableVariable(
                            VariableParameter {
                                value: ref value_x,
                                index: index_x,
                            },
                            VariableParameter {
                                value: ref value_y,
                                index: index_y,
                            },
                        ),
                    reverse,
                } => {
                    let (partial_x, partial_y) = reverse(&gradients[i], value_x, value_y);
                    gradients[index_x] = &gradients[index_x] + partial_x;
                    gradients[index_y] = &gradients[index_y] + partial_y;
                }
            }
        }
        Self(gradients)
    }

    /// Returns the gradient of the root variable with respect to the provided variable
    #[must_use]
    #[inline]
    pub fn wrt<const B: u64, const L: u64, const R: u64, const C: u64>(
        &self,
        x: &Variable<B, L, R, C>,
    ) -> Constant<B, L, R, C> {
        Constant::new(Values::Custom(self.0[x.index()].clone()))
    }
}
