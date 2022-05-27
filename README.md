# [![Mushin](assets/mushin-logo.png)](https://github.com/c0dearm/mushin)

[![Crates](https://img.shields.io/crates/v/mushin.svg)](https://crates.io/crates/mushin)
[![Crates](https://img.shields.io/crates/d/mushin.svg)](https://crates.io/crates/mushin)
[![Docs](https://docs.rs/mushin/badge.svg)](https://docs.rs/mushin)
[![CI](https://github.com/c0dearm/mushin/workflows/CI/badge.svg?branch=main)](https://github.com/c0dearm/mushin/actions)
[![Security](https://github.com/c0dearm/mushin/workflows/Security/badge.svg?branch=main)](https://github.com/c0dearm/mushin/actions)
[![Codecov](https://codecov.io/gh/c0dearm/mushin/branch/main/graph/badge.svg)](https://codecov.io/gh/c0dearm/mushin)
[![License](https://camo.githubusercontent.com/47069b7e06b64b608c692a8a7f40bc6915cf629c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d417061636865322e302532464d49542d626c75652e737667)](https://github.com/c0dearm/mushin/blob/master/COPYRIGHT)

[Mushin](https://en.wikipedia.org/wiki/Mushin_(mental_state)) is a Japanese term used in martial arts that refers to the state of mind obtained by practice. At this point, a person relies not on what they think should be the next move, but what is their trained natural reaction (or instinct).

## Description

**Mushin** is to `Rust` what `Tensorflow` is to `Python`. A library to build computational graphs and compute the gradients of outputs with respect to a given set of variables by applying [reverse automatic differentatiation](https://en.wikipedia.org/wiki/Automatic_differentiation).

Internally it uses the [arrayfire](https://crates.io/crates/arrayfire) crate to provide parallel computations on specialized hardware, such as Nvidia CUDA GPUs, Intel MKL CPUs... For details on what devices are available and installation instructions for your OS, please take a look at the `arrayfire` crate documentation. **The installation of the `arrayfire` binaries is required for `Mushin` to work.**

One clear benefit of this crate versus `Tensorflow` is Rust's strong type system. All operations performed on tensors during the graph build are checked at compile time for mathematical soundness, which means no runtime error after an hour of model training. **If it compiles, it works**. If at some point you make a mistake while building your made in hell nested computational graph, like for example on the shape of a tensor, you'll be stopped even before you can start feeling stupid.

Moreover, because constant and variable tensors are actually different types, the developer continuously has an overview of which resulting tensors contribute to the gradients and which not. What's more, the compiler will stop you from trying to compute the gradient of or with respect to a constant!

## Usage

First, install the arrayfire binaries as indicated by the [arrayfire](https://crates.io/crates/arrayfire) crate.

Then, add **Mushin** as one of your dependencies:

```toml
[dependencies]
mushin = "0.3"
```

The following is quite a self-explanatory example of the basic usage of **Mushin**, for more details, please check the crate [docs](https://docs.rs/mushin/latest/mushin/) or just ask us questions in the form of [issues](https://github.com/c0dearm/mushin/issues/new)! 😊

```rust
use mushin::{Context, Values, Gradients, add, matmul};

fn main() {
    let ctx = Context::new();

    let x = ctx.constant::<1, 1, 2, 3>(Values::Eye(3.0));
    let w = ctx.variable::<1, 1, 3, 2>(Values::Normal, Some("weights"));
    let b = ctx.variable::<1, 1, 3, 3>(Values::Fill(0.0), Some("bias"));
    let z = add(&b, &matmul(&w, &x));

    let grads = Gradients::compute(&z);
    let dz_dw = grads.wrt(&w);
    let dz_db = grads.wrt(&b);
}
```

## Roadmap

- [ ] Add more operations
- [ ] Allow for higher-order gradients
- [ ] Add benchmarks
- [ ] Add a cargo feature for deep learning, which adds layers, losses and activation functions (like `Keras`)

## Contributing

If you find a vulnerability, bug or would like a new feature, [open a new issue](https://github.com/c0dearm/mushin/issues/new).

To introduce your changes into the codebase, submit a Pull Request.

Many thanks!

## License

Mushin is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT), and
[COPYRIGHT](COPYRIGHT) for details.