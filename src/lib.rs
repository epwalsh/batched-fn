//! `batched-fn` provides a macro that can be used to easily a wrap a function that runs on
//! batches of inputs in such a way that it can be called with
//! a single input, yet where that single input is run as part of a batch of other inputs behind
//! the scenes.
//!
//! This is useful when you have a high throughput application where processing inputs in a batch
//! is more efficient that processing inputs one-by-one. The trade-off  is a small delay that is incurred
//! while waiting for a batch to be filled, though this can be tuned with the
//! [`delay`](macro.batched_fn.html#delay) and [`max_batch_size`](macro.batched_fn.html#max_batch_size)
//! parameters.
//!
//! A typical use-case is when you have a GPU-backed deep learning model deployed on a webserver that provides
//! a prediction for each input that comes in through an HTTP request.
//!
//! Even though inputs come individually - and outputs need to be served back individually - it
//! is usually more efficient to process a group of inputs together in order to fully utilize the GPU.
//!
//! In this case the model API might look like this:
//!
//! ```rust
//! // `Batch` could be anything that implements the `batched_fn::Batch` trait.
//! type Batch<T> = Vec<T>;
//!
//! #[derive(Debug)]
//! struct Input {
//!     // ...
//! }
//!
//! #[derive(Debug)]
//! struct Output {
//!     // ...
//! }
//!
//! struct Model {
//!     // ...
//! }
//!
//! impl Model {
//!     fn predict(&self, batch: Batch<Input>) -> Batch<Output> {
//!         // ...
//!         # batch.iter().map(|_| Output {}).collect()
//!     }
//!
//!     fn load() -> Self {
//!         // ...
//!         # Self {}
//!     }
//! }
//! ```
//!
//! Without `batched-fn`, the webserver route would need to call `Model::predict` on each
//! individual input which would result in a bottleneck from under-utilizing the GPU:
//!
//! ```rust
//! use once_cell::sync::Lazy;
//! # use batched_fn::{batched_fn, Batch as BatchTrait};
//! # type Batch<T> = Vec<T>;
//! # #[derive(Debug)]
//! # struct Input {}
//! # #[derive(Debug)]
//! # struct Output {}
//! # struct Model {}
//! # impl Model {
//! #     fn predict(&self, batch: Batch<Input>) -> Batch<Output> {
//! #         batch.iter().map(|_| Output {}).collect()
//! #     }
//! #     fn load() -> Self { Self {} }
//! # }
//! static MODEL: Lazy<Model> = Lazy::new(Model::load);
//!
//! fn predict_for_http_request(input: Input) -> Output {
//!     let mut batched_input = Batch::with_capacity(1);
//!     batched_input.push(input);
//!     MODEL.predict(batched_input).pop().unwrap()
//! }
//! ```
//!
//! But by dropping the [`batched_fn`](macro.batched_fn.html) macro into this function, you automatically get batched
//! inference behind the scenes without changing the one-to-one relationship between inputs and
//! outputs:
//!
//! ```rust
//! # use batched_fn::{batched_fn, Batch as BatchTrait};
//! # type Batch<T> = Vec<T>;
//! # #[derive(Debug)]
//! # struct Input {}
//! # #[derive(Debug)]
//! # struct Output {}
//! # struct Model {}
//! # impl Model {
//! #     fn predict(&self, batch: Batch<Input>) -> Batch<Output> {
//! #         batch.iter().map(|_| Output {}).collect()
//! #     }
//! #     fn load() -> Self { Self {} }
//! # }
//! async fn predict_for_http_request(input: Input) -> Output {
//!     let batch_predict = batched_fn! {
//!         handler = |batch: Batch<Input>, model: &Model| -> Batch<Output> {
//!             model.predict(batch)
//!         };
//!         config = {
//!             max_batch_size: 16,
//!             delay: 50,
//!         };
//!         context = {
//!             model: Model::load(),
//!         };
//!     };
//!     batch_predict(input).await
//! }
//! ```
//!
//! ❗️ *Note that the `predict_for_http_request` function now has to be `async`.*
//!
//! Here we set the [`max_batch_size`](macro.batch.html#max_batch_size) to 16 and [`delay`](macro.batched_fn.html#delay)
//! to 50 milliseconds. This means the batched function will wait at most 50 milliseconds after receiving a single
//! input to fill a batch of 16. If 15 more inputs are not received within 50 milliseconds
//! then the partial batch will be ran as-is.
//!
//! # Tuning max batch size and delay
//!
//! The optimal batch size and delay will depend on the specifics of your use case, such as how big of a batch you can fit in memory
//! (typically on the order of 8, 16, 32, or 64 for a deep learning model) and how long of a delay you can afford.
//! In general you want to set both of these as high as you can.
//!
//! It's worth noting that the response time of your application might actually go *down* under high load.
//! This is because the batch handler will be called as soon as either a batch of `max_batch_size` is filled or `delay` milliseconds
//! has passed, whichever happens first.
//! So under high load batches will be filled quickly, but under low load the response time will be at least `delay` milliseconds (adding the time
//! it takes to actually process a batch and respond).

extern crate once_cell;
extern crate tokio;

#[doc(hidden)]
pub use once_cell::sync::Lazy;

#[doc(hidden)]
pub use tokio::sync::{mpsc::UnboundedSender, Mutex};

use std::sync::mpsc::Sender;
use tokio::sync::mpsc as async_mpsc;

/// The `Batch` trait is essentially an abstraction of `Vec<T>`. The input and output of a batch
/// `handler` must implement `Batch`.
///
/// It represents an owned collection of ordered items of a single type.
pub trait Batch: IntoIterator<Item = <Self as Batch>::Item> {
    type Item;

    fn with_capacity(n: usize) -> Self;

    fn len(&self) -> usize;

    fn push(&mut self, item: <Self as Batch>::Item);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Batch for Vec<T> {
    type Item = T;

    fn with_capacity(n: usize) -> Vec<T> {
        Vec::<T>::with_capacity(n)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn push(&mut self, item: T) {
        self.push(item);
    }
}

#[doc(hidden)]
#[derive(Default)]
pub struct EmptyContext {}

/// A `BatchedFn` is a wrapper around a `handler` that provides the interface for
/// evaluating a single input as part of a batch of other inputs.
#[doc(hidden)]
pub struct BatchedFn<T, R>
where
    T: 'static + Send + Sync + std::fmt::Debug,
    R: 'static + Send + Sync + std::fmt::Debug,
{
    tx: Mutex<Sender<(T, UnboundedSender<R>)>>,
}

impl<T, R> BatchedFn<T, R>
where
    T: 'static + Send + Sync + std::fmt::Debug,
    R: 'static + Send + Sync + std::fmt::Debug,
{
    pub fn new(tx: Sender<(T, UnboundedSender<R>)>) -> Self {
        Self { tx: Mutex::new(tx) }
    }
    /// Evaluate a single input as part of a batch of other inputs.
    pub async fn evaluate_in_batch(&self, input: T) -> R {
        let (result_tx, mut result_rx) = async_mpsc::unbounded_channel::<R>();
        self.tx.lock().await.send((input, result_tx)).unwrap();
        result_rx.recv().await.unwrap()
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! __batch_fn_internal {
    (
        handler = |$batch:ident: $batch_input_type:ty $(, $ctx_arg:ident: &$ctx_arg_ty:ty )*| -> $batch_output_type:ty $fn_body:block ;
        config = {
            max_batch_size: $max_batch_size:expr,
            delay: $delay:expr $(,)?
        };
        context = {
            $( $ctx:ident: $ctx_init:expr ),* $(,)?
        };
    ) => {{
        static BATCHED_FN: $crate::Lazy<
            $crate::BatchedFn<
                <$batch_input_type as $crate::Batch>::Item,
                <$batch_output_type as $crate::Batch>::Item,
            >,
        > = $crate::Lazy::new(|| {
            let (tx, mut rx) = std::sync::mpsc::channel::<(
                <$batch_input_type as $crate::Batch>::Item,
                $crate::UnboundedSender<<$batch_output_type as $crate::Batch>::Item>,
            )>();

            std::thread::spawn(move || {
                // Create handler closure.
                let handler = |$batch: $batch_input_type $(, $ctx_arg: &$ctx_arg_ty )*| -> $batch_output_type {
                    $fn_body
                };

                // Set config vars.
                let max_batch_size: usize = $max_batch_size;
                let delay: u128 = $delay;

                // Initialize handler context.
                struct Context {
                    $( $ctx_arg: $ctx_arg_ty, )*
                }

                let context = Context {
                    $( $ctx: $ctx_init, )*
                };

                // Wait for an input.
                while let Ok((input, result_tx)) = rx.recv() {
                    let mut batch_input =
                        <$batch_input_type as $crate::Batch>::with_capacity(max_batch_size);
                    let mut batch_txs = Vec::with_capacity(max_batch_size);
                    batch_input.push(input);
                    batch_txs.push(result_tx);

                    let mut vacancy = max_batch_size - 1;
                    let mut time_left = delay as u64;
                    let start = std::time::Instant::now();

                    // While there is still room in the batch we'll wait at most `delay`
                    // milliseconds to try to fill it.
                    while vacancy > 0 && time_left > 0 {
                        if let Ok((next_input, next_result_tx)) =
                            rx.recv_timeout(std::time::Duration::from_millis(time_left))
                        {
                            batch_input.push(next_input);
                            batch_txs.push(next_result_tx);
                            vacancy -= 1;
                            let elapsed = start.elapsed().as_millis();
                            time_left = if elapsed > delay {
                                0
                            } else {
                                (delay - elapsed) as u64
                            };
                        } else {
                            break;
                        }
                    }

                    let batch_output = handler(batch_input $(, &context.$ctx_arg )*);
                    for (output, mut result_tx) in batch_output.into_iter().zip(batch_txs) {
                        result_tx.send(output).unwrap();
                    }
                }
            });

            $crate::BatchedFn::new(tx)
        });

        |input| BATCHED_FN.evaluate_in_batch(input)
    }};
}

/// Macro for creating a batched function.
///
/// This macro has 3 parameters: a `handler`, a `config`, and `context`.
///
/// ## `handler`
///
/// The handler must be in the form of a closure declaration that takes a batch
/// and any number of references to objects in the context as input and
/// outputs a different type of batch.
///
/// ### `config`
///
/// Within the config you must specify the `delay` and `max_batch_size`.
///
/// ### `delay`
///
/// This is the maximum number of milliseconds to wait for a batch to be filled after receiving
/// a single input.
///
/// ### `max_batch_size`
///
/// This is the maximum batch size that will be passed to the batch `handler`. When a batch
/// of this size is not filled before `delay` milliseconds the partial batch will be sent to the handler as-is.
///
/// ## `context`
///
/// Any additional reference that the handler takes as input must be defined within
/// the context.
#[macro_export]
macro_rules! batched_fn {
    (
        handler = |$batch:ident: $batch_input_type:ty $(, $ctx_arg:ident: &$ctx_arg_ty:ty )*| -> $batch_output_type:ty $fn_body:block ;
        config = {
            max_batch_size: $max_batch_size:expr,
            delay: $delay:expr $(,)?
        };
        context = {
            $( $ctx:ident: $ctx_init:expr ),* $(,)?
        } $(;)?
    ) => {
        $crate::__batch_fn_internal!(
            handler = |$batch: $batch_input_type $(, $ctx_arg: &$ctx_arg_ty )*| -> $batch_output_type $fn_body;
            config = {
                max_batch_size: $max_batch_size,
                delay: $delay,
            };
            context = {
                $( $ctx: $ctx_init ),*
            };
        );
    };
    (
        handler = |$batch:ident: $batch_input_type:ty $(, $ctx_arg:ident: &$ctx_arg_ty:ty )*| -> $batch_output_type:ty $fn_body:block ;
        config = {
            delay: $delay:expr,
            max_batch_size: $max_batch_size:expr $(,)?
        };
        context = {
            $( $ctx:ident: $ctx_init:expr ),* $(,)?
        } $(;)?
    ) => {
        $crate::__batch_fn_internal!(
            handler = |$batch: $batch_input_type $(, $ctx_arg: &$ctx_arg_ty )*| -> $batch_output_type $fn_body;
            config = {
                max_batch_size: $max_batch_size,
                delay: $delay,
            };
            context = {
                $( $ctx: $ctx_init ),*
            };
        );
    };
}
