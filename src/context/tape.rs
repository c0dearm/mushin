use std::cell::{Ref, RefCell};

use super::function::{
    DoubleParamReverseFn, DoubleParameter, Function, SingleParamReverseFn, VariableParameter,
};

/// Tape (or Wengert list) that keeps track of all the functions evaluated within the `Context` (a computation graph).
pub struct Tape(RefCell<Vec<Function>>);

/// Index of a node (function) in the `Tape`
pub type NodeId = usize;

impl Tape {
    pub const fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    /// Restart the computation graph from scratch
    pub fn reset(&self) {
        self.0.borrow_mut().clear();
    }

    pub fn functions(&self) -> Ref<Vec<Function>> {
        self.0.borrow()
    }

    pub fn len(&self) -> usize {
        self.functions().len()
    }

    /// Push a new variable declaration into the computation graph
    pub fn push_nary(&self) -> NodeId {
        let index = self.len();
        self.0.borrow_mut().push(Function::Nary);
        index
    }

    /// Push a single parameter function into the computation graph
    pub fn push_unary(&self, param: VariableParameter, reverse: SingleParamReverseFn) -> NodeId {
        let index = self.len();
        self.0.borrow_mut().push(Function::Unary { param, reverse });
        index
    }

    /// Push a double parameter function into the computation graph
    pub fn push_binary(&self, params: DoubleParameter, reverse: DoubleParamReverseFn) -> NodeId {
        let index = self.len();
        self.0
            .borrow_mut()
            .push(Function::Binary { params, reverse });
        index
    }
}
