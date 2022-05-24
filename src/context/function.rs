use arrayfire::Array;

use crate::tensor::{Origin, Tensor};

/// Represents an argument to a function and its value comes from a variable tensor
pub struct VariableArg {
    value: Array<f32>,
    function: usize,
}

/// Represents an argument to a function but its value comes from a constant tensor
pub struct ConstantArg {
    value: Array<f32>,
}

/// Represents the arguments of a function with two arguments
pub enum DoubleArg {
    /// Both arguments are variables
    BothVariables(VariableArg, VariableArg),
    /// First argument is a constant, second is a variable
    ConstFirstArg(ConstantArg, VariableArg),
    /// First argument is a variable, second is a constant
    ConstSecondArg(VariableArg, ConstantArg),
}

impl DoubleArg {
    const fn values(&self) -> (Option<usize>, &Array<f32>, Option<usize>, &Array<f32>) {
        match *self {
            Self::BothVariables(
                VariableArg {
                    value: ref a,
                    function: fa,
                },
                VariableArg {
                    value: ref b,
                    function: fb,
                },
            ) => (Some(fa), a, Some(fb), b),
            Self::ConstFirstArg(
                ConstantArg { value: ref a },
                VariableArg {
                    value: ref b,
                    function: fb,
                },
            ) => (None, a, Some(fb), b),
            Self::ConstSecondArg(
                VariableArg {
                    value: ref a,
                    function: fa,
                },
                ConstantArg { value: ref b },
            ) => (Some(fa), a, None, b),
        }
    }
}

/// Type of the function performing the reverse pass on a single argument function
type OneArgBackwardFn = fn(df: &Array<f32>, arg: &Array<f32>) -> Array<f32>;
/// Type of the function performing the reverse pass on a double argument function
type TwoArgsBackwardFn =
    fn(df: &Array<f32>, arg_a: &Array<f32>, arg_b: &Array<f32>) -> (Array<f32>, Array<f32>);

/// `f(x) = cos(x)` if x is a variable
pub struct OneArg {
    arg: VariableArg,
    backward: OneArgBackwardFn,
}

impl OneArg {
    pub(crate) fn backward(&self, df: &Array<f32>) -> (usize, Array<f32>) {
        (self.arg.function, (self.backward)(df, &self.arg.value))
    }
}

/// `f(x, y) = x * y` if at least one of them is a variable
pub struct TwoArgs {
    args: DoubleArg,
    backward: TwoArgsBackwardFn,
}

impl TwoArgs {
    pub(crate) fn backward(
        &self,
        df: &Array<f32>,
    ) -> (Option<usize>, Array<f32>, Option<usize>, Array<f32>) {
        let (f_a, arg_a, f_b, arg_b) = self.args.values();
        let (partial_a, partial_b) = (self.backward)(df, arg_a, arg_b);
        (f_a, partial_a, f_b, partial_b)
    }
}

/// Represents a node in the computation graph.
pub enum Function {
    /// A variable declaration (constants are ignored)
    Nary,
    /// A function with only one arg (constants are ignored), like `cos(x)`
    Unary(OneArg),
    /// A function with two args if at least one of them is a variable, like `x * y`
    Binary(TwoArgs),
}

impl Function {
    /// Creates a single argument function and pushes it to the tape, if the argument is a variable
    /// Returns a new tensor origin with a reference to the newly created function
    pub(crate) fn unary<const B: u64, const L: u64, const R: u64, const C: u64>(
        arg: &Tensor<B, L, R, C>,
        backward: OneArgBackwardFn,
    ) -> Origin {
        if let &Origin::Function(function) = arg.origin() {
            let function = Self::Unary(OneArg {
                arg: VariableArg {
                    value: arg.into(),
                    function,
                },
                backward,
            });
            Origin::Function(arg.context().push_function(function))
        } else {
            // Single argument function applied to a constant is a constant
            Origin::None
        }
    }

    /// Creates a double argument function and pushes it to the tape, if at least one of the arguments is a variable
    /// Returns a new tensor origin with a reference to the newly created function
    pub(crate) fn binary<
        const XB: u64,
        const XN: u64,
        const XR: u64,
        const XC: u64,
        const YB: u64,
        const YN: u64,
        const YR: u64,
        const YC: u64,
    >(
        arg_a: &Tensor<XB, XN, XR, XC>,
        arg_b: &Tensor<YB, YN, YR, YC>,
        backward: TwoArgsBackwardFn,
    ) -> Origin {
        match (arg_a.origin(), arg_b.origin()) {
            // If both arguments are a constant, result is a constant
            (&Origin::None, &Origin::None) => Origin::None,
            (&Origin::Function(function), &Origin::None) => {
                let function = Self::Binary(TwoArgs {
                    args: DoubleArg::ConstSecondArg(
                        VariableArg {
                            value: arg_a.into(),
                            function,
                        },
                        ConstantArg {
                            value: arg_b.into(),
                        },
                    ),
                    backward,
                });
                Origin::Function(arg_a.context().push_function(function))
            }
            (&Origin::None, &Origin::Function(function)) => {
                let function = Self::Binary(TwoArgs {
                    args: DoubleArg::ConstFirstArg(
                        ConstantArg {
                            value: arg_a.into(),
                        },
                        VariableArg {
                            value: arg_b.into(),
                            function,
                        },
                    ),
                    backward,
                });
                Origin::Function(arg_b.context().push_function(function))
            }
            (&Origin::Function(function_a), &Origin::Function(function_b)) => {
                let function = Self::Binary(TwoArgs {
                    args: DoubleArg::BothVariables(
                        VariableArg {
                            value: arg_a.into(),
                            function: function_a,
                        },
                        VariableArg {
                            value: arg_b.into(),
                            function: function_b,
                        },
                    ),
                    backward,
                });
                Origin::Function(arg_a.context().push_function(function))
            }
        }
    }
}
