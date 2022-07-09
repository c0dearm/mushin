# [![Mushin](assets/mushin-logo.svg)](https://github.com/c0dearm/mushin)

[![Crates](https://img.shields.io/crates/v/mushin.svg?style=for-the-badge)](https://crates.io/crates/mushin)
[![Downloads](https://img.shields.io/crates/d/mushin.svg?style=for-the-badge)](https://crates.io/crates/mushin)
[![Docs](https://img.shields.io/docsrs/mushin?style=for-the-badge)](https://docs.rs/mushin)
[![Build](https://img.shields.io/github/workflow/status/c0dearm/mushin/CI/main?style=for-the-badge)](https://github.com/c0dearm/mushin/actions)
[![Security](https://img.shields.io/github/workflow/status/c0dearm/mushin/Security/main?style=for-the-badge&label=Security)](https://github.com/c0dearm/mushin/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/c0dearm/mushin?style=for-the-badge)](https://codecov.io/gh/c0dearm/mushin)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack)](https://mushin-rs.slack.com)

[Mushin](https://en.wikipedia.org/wiki/Mushin_(mental_state)) is a Japanese term used in martial arts that refers to the state of mind obtained by practice. At this point, a person relies not on what they think should be the next move, but what is their trained natural reaction (or instinct).

## Description

**Mushin** is a pure `Rust`, no-unsafe library for computing gradients on dynamic computational graphs using [reverse automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation). In other words, what `PyTorch` is to `Python` is what `Mushin` is to `Rust`.

Internally it uses the [arrayfire](https://crates.io/crates/arrayfire) crate to provide parallel computations on specialized hardware, such as Nvidia CUDA GPUs, Intel MKL CPUs... For details on what devices are available and installation instructions for your OS, please take a look at the `arrayfire` crate documentation. **The installation of the `arrayfire` binaries is required for `Mushin` to work.**

One clear benefit of this crate versus `PyTorch` is `Rust`'s strong type system. All operations performed on tensors during the graph build are checked at compile time for mathematical soundness, which means no runtime error after an hour of model training. **If it compiles, it works**. If at some point you make a mistake while building your made in hell nested computational graph, like for example on the shape of a tensor, you'll be stopped even before you can start feeling stupid.

Moreover, because constant and variable tensors are actually different types, the developer continuously has an overview of which resulting tensors contribute to the gradients and which not. On top of that, the compiler will stop you from trying to compute the gradient of or with respect to a constant!

Another benefit when compared to other similar libraries is that the computation graph is eagerly evaluated, which means that the graph is **trully dynamic**. In other words, your next operations can be conditioned to the results of previous ones, and so you can have conditional branching while
building your graph.

## Usage

First, install the arrayfire binaries as indicated by the [arrayfire](https://crates.io/crates/arrayfire) crate.

Then, add `mushin` as one of your dependencies:

```toml
[dependencies]
mushin = "0.5"
```

The following is quite a self-explanatory example of the basic usage of **Mushin** to build computation graphs and get the derivatives back:
```rust
use mushin as mu;

fn main() {
    let x = mu::eye::<1, 1, 2, 3>(3.0).freeze();
    let w = mu::randn::<1, 1, 3, 2>();
    let b = mu::fill::<1, 1, 3, 3>(0.0);

    let z = mu::add(&mu::mm(&w, &x), &b);
    z.backward();

    let dz_dw = w.grad()
    let dz_db = b.grad()
}
```

By default, this library enables the `nn` feature that gives access to the `nn` module, which builds upon the auto-grad foundation of `Mushin` to deliver a set of **Deep Learning** utilities, such as activation functions, layers, losses and optimizers. If you don't really need that part and you are only insterested in the pure auto-grad functionality of this library, the `nn` module can be disabled with `default-features = false`. Here follows a brief example on how it works:

```rust
use mushin as mu;
use mu::nn::{layers::Linear, activations::relu, losses::mse, optimizers::SGD};

let x = mu::eye::<16, 1, 1, 3>(1.0).freeze();
let y = mu::eye::<16, 1, 1, 5>(3.0).freeze();

let linear = Linear::<16, 3, 5>::new();
let optim = SGD::new(&[linear.parameters()], 0.01);

for _ in 0..5 {
    let z = relu(&linear.forward(&x));
    let loss = mse(&z, &y);
    
    loss.backward();
    optim.step();
    loss.reset();
}
```

## Contributing

* If you find a vulnerability, bug or miss something, please [open a new issue](https://github.com/c0dearm/mushin/issues/new)
* To introduce your changes into the codebase, submit a [pull request](https://github.com/c0dearm/mushin/pulls)
* To discuss about possible improvements, suggestions and new fearures, [join us in Slack!](https://mushin-rs.slack.com)

Many thanks!

## License

Mushin is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT), and
[COPYRIGHT](COPYRIGHT) for details.