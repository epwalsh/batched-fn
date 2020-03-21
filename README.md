<div align="center">
    <h1>batched-fn</h1>
    An experimental Rust macro for creating batched functions that can be called with a single input.
</div>
<br/>
<p align="center">
    <a href="https://github.com/epwalsh/batched-fn/actions">
        <img alt="Build" src="https://github.com/epwalsh/batched-fn/workflows/CI/badge.svg?event=push&branch=master">
    </a>
    <a href="https://github.com/epwalsh/batched-fn/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/epwalsh/batched-fn.svg?color=blue&cachedrop">
    </a>
    <a href="https://crates.io/crates/batched-fn">
        <img alt="Crates" src="https://img.shields.io/crates/v/batched-fn.svg?color=blue">
    </a>
    <a href="https://docs.rs/batched-fn/">
        <img alt="Docs" src="https://img.shields.io/badge/docs.rs-API%20docs-blue">
    </a>
</p>
<br/>

`batched-fn` provides a macro that can be used to easily a wrap a function that runs on
batches of inputs in such a way that it can be called with
a single input, yet where that single input is run as part of a batch of other inputs behind
the scenes. This is useful when you have a high throughput application where processing inputs in a batch
is more efficient that processing inputs one-by-one.

Consider, for example, a GPU-backed deep learning model deployed on a webserver that provides
a prediction for each input that comes in through an HTTP request. Even though inputs come
individually - and outputs need to be served back individually - it is more efficient to process
a group of inputs together in order to fully utilize the GPU.

Here `batched-fn` could be dropped in to allow batched processing of inputs without changing
the one-to-one relationship of inputs to outputs. The trade-off  is a small delay that is incurred
while waiting for a batch to be filled, though this can be tuned with the `delay` parameter.

## Examples

⚠️ *These examples are obviously contrived and wouldn't be practical since there is no gain here
from processing inputs in a batch.*

Suppose we want a public API that allows a user to double a single number, but for some reason
its more efficient to double a batch of numbers together. Without batching we could just have
a simple function like this:

```rust
fn double(x: i32) -> i32 {
    x * 2
}
```

But adding batching behind the scenes is as simple as dropping the `batched_fn` macro
in the function body:

```rust
async fn double(x: i32) -> i32 {
    let batched_double = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> {
            batch.iter().map(|batch_i| batch_i * 2).collect()
        }
    };
    batched_double(x).await
}
```

❗️ *Note that the `double` function now has to be `async`.*

We can also adjust the maximum batch size and tune the wait delay.

Here we set the `max_batch_size` to 4 and `delay` to 50 milliseconds. This means
the batched function will wait at most 50 milliseconds after receiving a single
input to fill a batch of 4. If 3 more inputs are not received within 50 milliseconds
then the partial batch will be ran as-is.

```rust
async fn double(x: i32) -> i32 {
    let batched_double = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> {
            batch.iter().map(|batch_i| batch_i * 2).collect()
        },
        max_batch_size = 4,
        delay = 50,
    };
    batched_double(x).await
}
```
