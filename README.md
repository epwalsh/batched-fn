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
the scenes.

This is useful when you have a high throughput application where processing inputs in a batch
is more efficient that processing inputs one-by-one. The trade-off  is a small delay that is incurred
while waiting for a batch to be filled, though this can be tuned with the
`delay` and `max_batch_size` parameters.

A typical use-case is when you have a GPU-backed deep learning model deployed on a webserver that provides
a prediction for each input that comes in through an HTTP request.

Even though inputs come individually - and outputs need to be served back individually - it
is usually more efficient to process a group of inputs together in order to fully utilize the GPU.

In this case the model API might look like this:

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

Without `batched-fn`, the webserver route would need to call `Model::predict` on each
individual input which would result in a bottleneck from under-utilizing the GPU:

```rust
use once_cell::sync::Lazy;

static MODEL: Lazy<Model> = Lazy::new(Model::load);

fn predict_for_http_request(input: Input) -> Output {
    let mut batched_input = Batch::with_capacity(1);
    batched_input.push(input);
    MODEL.predict(batched_input).pop().unwrap()
}
```

But by dropping the `batched_fn` macro into this function, you automatically get batched
inference behind the scenes without changing the one-to-one relationship between inputs and
outputs:

```rust
async fn predict_for_http_request(input: Input) -> Output {
    let batch_predict = batched_fn! {
        handler = |batch: Batch<Input>, model: &Model| -> Batch<Output> {
            ctx.model.predict(batch)
        };
        config = {
            max_batch_size: 16,
            delay: 50,
        };
        context = {
            model = Model::load(),
        };
    };
    batch_predict(input).await
}
```

❗️ *Note that the `predict_for_http_request` function now has to be `async`.*

Here we set the `max_batch_size` to 16 and `delay`
to 50 milliseconds. This means the batched function will wait at most 50 milliseconds after receiving a single
input to fill a batch of 16. If 15 more inputs are not received within 50 milliseconds
then the partial batch will be ran as-is.

## Tuning max batch size and delay

The optimal batch size and delay will depend on the specifics of your use case, such as how big of a batch you can fit in memory
(typically on the order of 8, 16, 32, or 64 for a deep learning model) and how long of a delay you can afford.
In general you want to set both of these as high as you can.

It's worth noting that the response time of your application might actually go *down* under high load.
This is because the batch handler will be called as soon as either a batch of `max_batch_size` is filled or `delay` milliseconds
has passed, whichever happens first.
So under high load batches will be filled quickly, but under low load the response time will be at least `delay` milliseconds (adding the time
it takes to actually process a batch and respond).
