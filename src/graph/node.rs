use arrayfire::{constant, Array};
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Type used to as `Node` identifier
#[allow(clippy::module_name_repetitions)]
pub type NodeId = usize;

/// Represents the origin of a `Node`
enum Origin {
    /// The node is a new variable declaration
    Declaration,
    /// The node is the result of a unary operation, like `sin(x)`
    Unary(UnaryOp),
    /// The node is the result of a binary operation, like `x + y`
    Binary(BinaryOp),
}

/// A `Node` holds a `Variable` tensor data (values and gradients) as
/// well as information about its `Origin`
pub struct Node {
    id: NodeId,
    data: RefCell<Array<f32>>,
    grad: RefCell<Array<f32>>,
    origin: Origin,
}

impl Node {
    /// Creates a new `Node` with the given data and `Origin`. Gradients
    /// are set to zero by default. Each new `Node` has a unique ID fetched
    /// from a global static incremental counter. Unique IDs are necessary
    /// to be able to tell if two nodes (tensors) are the same when used in
    /// different operations.
    fn new(data: Array<f32>, origin: Origin) -> Self {
        let dims = data.dims();

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(constant(0.0, dims)),
            origin,
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Creates a new `Node` with a declaration `Operation` as origin
    pub(crate) fn declaration(data: Array<f32>) -> Self {
        Self::new(data, Origin::Declaration)
    }

    /// Creates a new `Node` with a unary `Operation` as origin
    pub(crate) fn unary(
        data: Array<f32>,
        ancestor: Rc<Self>,
        reverse: UnaryReverseFn,
        args: &[Array<f32>],
    ) -> Self {
        Self::new(
            data,
            Origin::Unary(UnaryOp {
                ancestor,
                reverse,
                args: args.to_vec(),
            }),
        )
    }

    /// Creates a new `Node` with a binary `Operation` as origin and both operation
    /// parameters are `Variable`s
    pub(crate) fn binary_varvar(
        data: Array<f32>,
        ancestors: (Rc<Self>, Rc<Self>),
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Self {
        Self::new(
            data,
            Origin::Binary(BinaryOp {
                ancestors: BinaryParams::VarVar(ancestors.0, ancestors.1),
                reverse,
                args: args.to_vec(),
            }),
        )
    }

    /// Creates a new `Node` with a binary `Operation` as origin and only the
    /// first operation parameter is a `Variable`
    pub(crate) fn binary_varconst(
        data: Array<f32>,
        ancestor: Rc<Self>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Self {
        Self::new(
            data,
            Origin::Binary(BinaryOp {
                ancestors: BinaryParams::VarConst(ancestor),
                reverse,
                args: args.to_vec(),
            }),
        )
    }

    /// Creates a new `Node` with a binary `Operation` as origin and only the
    /// second operation parameter is a `Variable`
    pub(crate) fn binary_constvar(
        data: Array<f32>,
        ancestor: Rc<Self>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Self {
        Self::new(
            data,
            Origin::Binary(BinaryOp {
                ancestors: BinaryParams::ConstVar(ancestor),
                reverse,
                args: args.to_vec(),
            }),
        )
    }

    /// Returns the tensor data
    pub(crate) fn data(&self) -> Ref<Array<f32>> {
        self.data.borrow()
    }

    /// Returns a mutable reference to the tensor data
    pub(crate) fn data_mut(&self) -> RefMut<Array<f32>> {
        self.data.borrow_mut()
    }

    /// Returns the tensor gradients
    pub(crate) fn grad(&self) -> Ref<Array<f32>> {
        self.grad.borrow()
    }

    /// Returns a mutable reference to the tensor gradients
    pub(crate) fn grad_mut(&self) -> RefMut<Array<f32>> {
        self.grad.borrow_mut()
    }

    /// Computes the gradients of this node ancestors by following the
    /// computation graph backwards
    pub(crate) fn reverse(&self) {
        match self.origin {
            Origin::Unary(ref op) => {
                op.reverse(&self.grad());
            }
            Origin::Binary(ref op) => {
                op.reverse(&self.grad());
            }
            Origin::Declaration => {}
        }
    }

    /// Sets all its gradient values to one
    pub(crate) fn ones_grad(&self) {
        let dims = self.grad().dims();
        *self.grad_mut() = constant(1.0, dims);
    }

    /// Sets all its gradient values to zero
    pub(crate) fn zero_grad(&self) {
        let dims = self.grad().dims();
        *self.grad_mut() = constant(0.0, dims);
    }

    /// Returns node's ID
    pub(crate) const fn id(&self) -> NodeId {
        self.id
    }

    /// Returns `true` if the node is `Variable` declaration, `false` otherwise
    pub(crate) const fn is_declaration(&self) -> bool {
        matches!(self.origin, Origin::Declaration)
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        COUNTER.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Represents the different combination of parameters a binary `Operation`
/// can have
enum BinaryParams {
    /// Both parameters are `Variable`s
    VarVar(Rc<Node>, Rc<Node>),
    /// Only one parameter is a `Variable`
    VarConst(Rc<Node>),
    /// Only the second parameter is a `Variable`
    ConstVar(Rc<Node>),
}

/// Computes the partial adjoint derivative of a unary operation for its parameter
pub type UnaryReverseFn = fn(df: &Array<f32>, args: &[Array<f32>]) -> Array<f32>;
/// Computes the partial adjoint derivative of a binary operation for each of its parameters
pub type BinaryReverseFn = fn(df: &Array<f32>, args: &[Array<f32>]) -> (Array<f32>, Array<f32>);

/// Represents a unary `Operation`
struct UnaryOp {
    ancestor: Rc<Node>,
    reverse: UnaryReverseFn,
    args: Vec<Array<f32>>,
}

impl UnaryOp {
    /// Computes the partial adjoint derivative and accumulates it to the parameter gradients
    fn reverse(&self, df: &Array<f32>) {
        let partial = (self.reverse)(df, self.args.as_slice());
        *self.ancestor.grad_mut() += partial;
    }
}

/// Represents a binary `Operation`
struct BinaryOp {
    ancestors: BinaryParams,
    reverse: BinaryReverseFn,
    args: Vec<Array<f32>>,
}

impl BinaryOp {
    /// Computes the partial adjoints derivatives and accumulates them to the parameters gradients
    fn reverse(&self, df: &Array<f32>) {
        match self.ancestors {
            BinaryParams::VarVar(ref ancestor_a, ref ancestor_b) => {
                let (partial_a, partial_b) = (self.reverse)(df, self.args.as_slice());
                let grad_a = arrayfire::add(&ancestor_a.grad().clone(), &partial_a, true);
                let grad_b = arrayfire::add(&ancestor_b.grad().clone(), &partial_b, true);
                *ancestor_a.grad_mut() = grad_a;
                *ancestor_b.grad_mut() = grad_b;
            }
            BinaryParams::VarConst(ref ancestor) => {
                let (partial, _) = (self.reverse)(df, self.args.as_slice());
                let grad = arrayfire::add(&ancestor.grad().clone(), &partial, true);
                *ancestor.grad_mut() = grad;
            }
            BinaryParams::ConstVar(ref ancestor) => {
                let (_, partial) = (self.reverse)(df, self.args.as_slice());
                let grad = arrayfire::add(&ancestor.grad().clone(), &partial, true);
                *ancestor.grad_mut() = grad;
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{Node, Origin};
    use crate::tests::equal_arrays;

    #[test]
    fn new_node() {
        let node = Node::new(arrayfire::constant!(2.0; 1,2,3,4), Origin::Declaration);
        assert!(equal_arrays(
            node.data().clone(),
            arrayfire::constant!(2.0; 1,2,3,4)
        ));
        assert!(equal_arrays(
            node.grad().clone(),
            arrayfire::constant!(0.0; 1,2,3,4)
        ));
        assert!(matches!(node.origin, Origin::Declaration));
        assert_eq!(node.id(), 0);
    }

    #[test]
    fn node_sequentially_reused_unique_ids() {
        let node = Node::new(arrayfire::constant!(2.0; 1,2,3,4), Origin::Declaration);
        assert_eq!(node.id(), 0);

        let node = Node::new(arrayfire::constant!(2.0; 1,2,3,4), Origin::Declaration);
        assert_eq!(node.id(), 1);

        {
            let node = Node::new(arrayfire::constant!(2.0; 1,2,3,4), Origin::Declaration);
            assert_eq!(node.id(), 2);
        }

        // Node 2 is dropped and its ID is reused
        let node = Node::new(arrayfire::constant!(2.0; 1,2,3,4), Origin::Declaration);
        assert_eq!(node.id(), 2);

        let node = Node::new(arrayfire::constant!(2.0; 1,2,3,4), Origin::Declaration);
        assert_eq!(node.id(), 3);
    }

    #[test]
    fn ones_grad() {
        let node = Node::new(arrayfire::constant!(2.0; 1,2,3,4), Origin::Declaration);
        node.ones_grad();
        assert!(equal_arrays(
            node.grad().clone(),
            arrayfire::constant!(1.0; 1,2,3,4)
        ));
    }

    #[test]
    fn zero_grad() {
        let node = Node::new(arrayfire::constant!(2.0; 1,2,3,4), Origin::Declaration);
        node.zero_grad();
        assert!(equal_arrays(
            node.grad().clone(),
            arrayfire::constant!(0.0; 1,2,3,4)
        ));
    }
}
