use crate::graph::node::{Node, NodeId};
use std::collections::{btree_map::Values, BTreeMap};
use std::rc::Rc;

/// Stores the computation graph as a set of operation `Node`s
#[derive(Default, Clone)]
pub struct Tape(BTreeMap<NodeId, Rc<Node>>);

impl Tape {
    /// Adds the node to the computation graph
    pub(crate) fn push(&mut self, node: Rc<Node>) {
        self.0.insert(node.id(), node);
    }

    /// Return an iterator over the computation graph nodes
    pub(crate) fn nodes(&self) -> Values<NodeId, Rc<Node>> {
        self.0.values()
    }

    /// Given another tape, returns a new tape with the joined computation graphs
    pub(crate) fn merge(&self, other: &Self) -> Self {
        let mut tape = self.clone();
        let other = other.0.clone();
        tape.0.extend(other.into_iter());
        tape
    }
}

#[cfg(test)]
mod tests {
    use super::Tape;
    use crate::graph::node::Node;
    use std::rc::Rc;

    #[test]
    fn merge_tapes() {
        let mut first = Tape::default();
        let mut second = Tape::default();
        let node_0 = Rc::new(Node::declaration(arrayfire::constant!(1.0; 1,2,3,4)));
        let node_1 = Rc::new(Node::declaration(arrayfire::constant!(1.0; 1,2,3,4)));
        let node_2 = Rc::new(Node::declaration(arrayfire::constant!(1.0; 1,2,3,4)));

        first.push(node_0.clone());
        first.push(node_1.clone());
        second.push(node_1.clone());
        second.push(node_2.clone());

        let result = first.merge(&second);
        assert_eq!(result.0.len(), 3);

        for (i, (k, v)) in result.0.iter().enumerate() {
            assert_eq!(i, *k);
            assert_eq!(i, v.id());
        }
    }
}
