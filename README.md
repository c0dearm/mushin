# Mushin: Compile-time creation of neural networks

[![CI](https://github.com/c0dearm/mushin/workflows/CI/badge.svg?branch=main)](https://github.com/c0dearm/mushin/actions)
[![Security](https://github.com/c0dearm/mushin/workflows/Security/badge.svg?branch=main)](https://github.com/c0dearm/mushin/actions)
[![Crates](https://img.shields.io/crates/v/mushin.svg)](https://crates.io/crates/mushin)
[![Docs](https://docs.rs/mushin/badge.svg)](https://docs.rs/mushin)
[![Codecov](https://codecov.io/gh/c0dearm/mushin/branch/main/graph/badge.svg)](https://codecov.io/gh/c0dearm/mushin)
[![License](https://camo.githubusercontent.com/47069b7e06b64b608c692a8a7f40bc6915cf629c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d417061636865322e302532464d49542d626c75652e737667)](https://github.com/c0dearm/mushin/blob/master/COPYRIGHT)

[Mushin](https://en.wikipedia.org/wiki/Mushin_(mental_state)) is a Japanese term used in martial arts that refers to the state of mind obtained by practice. At this point, a person relies not on what they think should be the next move, but what is their trained natural reaction (or instinct).

## Description

Mushin allows the developer to build neural networks at compile-time, with preallocated arrays with well defined sizes. This has mainly three very important benefits:

1. **Compile-time network consistency check**: Any defect in your neural network (i.e. mismatching layers inputs/outputs) will be raised at compile-time. You can enjoy your coffee while your network inference or training process never fails!
2. **Awesome Rust compiler optimizations**: Because the neural network is completely defined at compile-time, the compiler is able
to perform smart optimizations, like unrolling loops or injecting [SIMD](https://en.wikipedia.org/wiki/SIMD) instructions.
3. **Support for embedded**: The `std` library is not required to build neural networks so it can run on any target that Rust supports.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
mushin = "0.1"
mushin_derive = "0.1"
```

And this is a very simple example to get you started:

```rust
use rand::distributions::Uniform;

use mushin::{activations::ReLu, layers::Dense, NeuralNetwork};
use mushin_derive::NeuralNetwork;

// Builds a neural network with 2 inputs and 1 output
// Made of 3 feed forward layers, you can have as many as you want and with any name
#[derive(NeuralNetwork, Debug)]
struct MyNetwork {
    // LayerType<ActivationType, # inputs, # outputs>
    input: Dense<ReLu, 2, 4>,
    hidden: Dense<ReLu, 4, 2>,
    output: Dense<ReLu, 2, 1>,
}

impl MyNetwork {
    // Initialize layer weights with a uniform distribution and set ReLU as activation function
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-1.0..=1.0);

        MyNetwork {
            input: Dense::random(&mut rng, &dist),
            hidden: Dense::random(&mut rng, &dist),
            output: Dense::random(&mut rng, &dist),
        }
    }
}

fn main() {
    // Init the weights and perform a forward pass
    let nn = MyNetwork::new();
    println!("{:#?}", nn);

    let input = [0.0, 1.0];
    println!("Input: {:#?}", input);
    let output = nn.forward(input);
    println!("Output: {:#?}", output);
}
```

You may wonder how the `forward` method works. The `NeuralNetwork` derive macro defines it for you, and it looks like this for this particular example:

```rust
fn forward(&self, input: [f32; 2]) -> [f32; 1] {
    self.output.forward(self.hidden.forward(self.input.forward[input]))
}
```

Note how the forward method expects two input values because that's what the first (`input`) layer expects, and returns one single value because that's what the last layer (`output`) returns.

## Roadmap

- [x] Compile-time neural network consistency check
- [x] Docs, CI/CD & Benchmarks
- [ ] Backward pass
- [ ] More layer types (convolution, dropout, lstm...)
- [ ] More activation functions (sigmoid, softmax...)
- [ ] Maaaybeee, CPU and/or GPU concurrency

## Contributing

If you find a vulnerability, bug or would like a new feature, [open a new issue](https://github.com/c0dearm/mushin/issues/new).

To introduce your changes into the codebase, submit a Pull Request.

Many thanks!

## License

Mushin is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT), and
[COPYRIGHT](COPYRIGHT) for details.
