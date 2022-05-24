use std::cell::{Ref, RefCell};

use crate::context::function::Function;

/// Tape (or Wengert list) that keeps track of all the expressions evaluated since its declaration (a computation graph).
/// Used by the `Context`
pub struct Tape(RefCell<Vec<Function>>);

impl Tape {
    pub(crate) const fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    pub(crate) fn reset(&self) {
        self.0.borrow_mut().clear();
    }

    pub(crate) fn functions(&self) -> Ref<Vec<Function>> {
        self.0.borrow()
    }

    pub(crate) fn len(&self) -> usize {
        self.0.borrow().len()
    }

    pub(crate) fn push_function(&self, function: Function) -> usize {
        let index = self.functions().len();
        self.0.borrow_mut().push(function);
        index
    }
}
