<div align="center">
    <h1>batched-fn</h1>
    Rust server plugin for deploying deep learning models with batched prediction.
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

<!--
DO NOT EDIT BELOW THIS POINT BY HAND!

Everything below this point is automatically generated using cargo-rdme: https://github.com/orium/cargo-rdme
Just run `make readme` to update.
-->

<!-- cargo-rdme start -->

Deep learning models are usually implemented to make efficient use of a GPU by batching inputs together
in "mini-batches". However, applications serving these models often receive requests one-by-one.
So using a conventional single or multi-threaded server approach will under-utilize the GPU and lead to latency that increases
linearly with the volume of requests.

`batched-fn` is a drop-in solution for deep learning webservers that queues individual requests and provides them as a batch
to your model. It can be added to any application with minimal refactoring simply by inserting the [`batched_fn`](https://docs.rs/batched-fn/latest/batched_fn/macro.batched_fn.html)
macro into the function that runs requests through the model.

## Features

- 🚀 Easy to use: drop the `batched_fn!` macro into existing code.
- 🔥 Lightweight and fast: queue system implemented on top of the blazingly fast [flume crate](https://github.com/zesterer/flume).
- 🙌 Easy to tune: simply adjust [`max_delay`](https://docs.rs/batched-fn/latest/batched_fn/macro.batched_fn.html#config) and [`max_batch_size`](https://docs.rs/batched-fn/latest/batched_fn/macro.batched_fn.html#config).
- 🛑 [Back pressure](https://medium.com/@jayphelps/backpressure-explained-the-flow-of-data-through-software-2350b3e77ce7) mechanism included:
  just set [`channel_cap`](https://docs.rs/batched-fn/latest/batched_fn/macro.batched_fn.html#config) and handle
  [`Error::Full`](https://docs.rs/batched-fn/latest/batched_fn/enum.Error.html#variant.Full) by returning a 503 from your webserver.

## Examples

Suppose you have a model API that look like this:

```rust
// `Batch` could be anything that implements the `batched_fn::Batch` trait.
type Batch<T> = Vec<T>;

#[derive(Debug)]
struct Input {
    // ...
}

#[derive(Debug)]
struct Output {
    // ...
}

struct Model {
    // ...
}

impl Model {
    fn predict(&self, batch: Batch<Input>) -> Batch<Output> {
        // ...
    }

    fn load() -> Self {
        // ...
    }
}
```

Without `batched-fn` a webserver route would need to call `Model::predict` on each
individual input, resulting in a bottleneck from under-utilizing the GPU:

```rust
use once_cell::sync::Lazy;
static MODEL: Lazy<Model> = Lazy::new(Model::load);

fn predict_for_http_request(input: Input) -> Output {
    let mut batched_input = Batch::with_capacity(1);
    batched_input.push(input);
    MODEL.predict(batched_input).pop().unwrap()
}
```

But by dropping the [`batched_fn`](https://docs.rs/batched-fn/latest/batched_fn/macro.batched_fn.html) macro into your code you automatically get batched
inference behind the scenes without changing the one-to-one relationship between inputs and
outputs:

```rust
async fn predict_for_http_request(input: Input) -> Output {
    let batch_predict = batched_fn! {
        handler = |batch: Batch<Input>, model: &Model| -> Batch<Output> {
            model.predict(batch)
        };
        config = {
            max_batch_size: 16,
            max_delay: 50,
        };
        context = {
            model: Model::load(),
        };
    };
    batch_predict(input).await.unwrap()
}
```

❗️ *Note that the `predict_for_http_request` function now has to be `async`.*

Here we set the [`max_batch_size`](https://docs.rs/batched-fn/latest/batched_fn/macro.batched_fn.html#config) to 16 and [`max_delay`](https://docs.rs/batched-fn/latest/batched_fn/macro.batched_fn.html#config)
to 50 milliseconds. This means the batched function will wait at most 50 milliseconds after receiving a single
input to fill a batch of 16. If 15 more inputs are not received within 50 milliseconds
then the partial batch will be ran as-is.

## Tuning max batch size and max delay

The optimal batch size and delay will depend on the specifics of your use case, such as how big of a batch you can fit in memory
(typically on the order of 8, 16, 32, or 64 for a deep learning model) and how long of a delay you can afford.
In general you want to set `max_batch_size` as high as you can, assuming the total processing time for `N` examples is minimized
with a batch size of `N`, and keep `max_delay` small relative to the time it takes for your
handler function to process a batch.

## Implementation details

When the `batched_fn` macro is invoked it spawns a new thread where the
[`handler`](https://docs.rs/batched-fn/latest/batched_fn/macro.batched_fn.html#handler) will
be ran. Within that thread, every object specified in the [`context`](https://docs.rs/batched-fn/latest/batched_fn/macro.batched_fn.html#context)
is initialized and then passed by reference to the handler each time it is run.

The object returned by the macro is just a closure that sends a single input and a callback
through an asyncronous channel to the handler thread. When the handler finishes
running a batch it invokes the callback corresponding to each input with the corresponding output,
which triggers the closure to wake up and return the output.

<!-- cargo-rdme end -->
