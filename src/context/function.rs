use arrayfire::Array;

use super::tape::NodeId;

/// Represents a variable single parameter
pub struct VariableParameter {
    pub value: Array<f32>,
    pub index: NodeId,
}

impl VariableParameter {
    pub fn new(value: Array<f32>, index: NodeId) -> Self {
        Self { value, index }
    }
}

/// Represents a single constant parameter
pub struct ConstantParameter {
    pub value: Array<f32>,
}

impl ConstantParameter {
    pub fn new(value: Array<f32>) -> Self {
        Self { value }
    }
}

/// Represents the possible combination of parameters for a binary function.
/// Note that binary functions with both constant parameters are not considered for the computation graph
/// because the result is also a constant and by definition it doesn't have a derivative
pub enum DoubleParameter {
    /// First parameters is a variable and second parameter is a constant
    VariableConstant(VariableParameter, ConstantParameter),
    /// First parameters is a constant and second parameter is a variable
    ConstantVariable(ConstantParameter, VariableParameter),
    /// Both parameters are variables
    VariableVariable(VariableParameter, VariableParameter),
}

/// Compute the derivative of a unary function with respect to its parameter (using the reverse chain rule `df(z)/dx = (df/dz)(dz/dx)`)
pub type SingleParamReverseFn = fn(&Array<f32>, &Array<f32>) -> Array<f32>;
/// Compute the derivative of a binary function with respect to each of its parameters (using the reverse chain rule `df(z,w)/dx = [(df/dz)(dz/dx), (df/dw)(dw/dx)]`)
pub type DoubleParamReverseFn =
    fn(&Array<f32>, &Array<f32>, &Array<f32>) -> (Array<f32>, Array<f32>);

/// Represents a node in the computation graph. Only functions whose result is a variable
/// are considered and introduced to the computation graph, derivatives of constants
/// or with respect to constants are mathematically undefined
pub enum Function {
    // A variable declaration
    Nary,
    // A function with a single variable parameter, like `f(x) = cos(x)`
    Unary {
        param: VariableParameter,
        reverse: SingleParamReverseFn,
    },
    // A function with two parameters where at least one of them is a variable, like `f(x,y) = x * y`
    Binary {
        params: DoubleParameter,
        reverse: DoubleParamReverseFn,
    },
}
