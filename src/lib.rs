//! `batched-fn` provides a macro that can be used to easily a wrap a function that runs on
//! batches of inputs in such a way that it can be called with
//! a single input, yet where that single input is run as part of a batch of other inputs behind
//! the scenes. This is useful when you have a high throughput application where processing inputs in a batch
//! is more efficient that processing inputs one-by-one.
//!
//! Consider, for example, a GPU-backed deep learning model deployed on a webserver that provides
//! a prediction for each input that comes in through an HTTP request. Even though inputs come
//! individually - and outputs need to be served back individually - it is more efficient to process
//! a group of inputs together in order to fully utilize the GPU.
//!
//! Here `batched-fn` could be dropped in to allow batched processing of inputs without changing
//! the one-to-one relationship of inputs to outputs. The trade-off  is a small delay that is incurred
//! while waiting for a batch to be filled, though this can be tuned with the `delay` parameter.
//!
//! # Examples
//!
//! ⚠️ *These examples are obviously contrived and wouldn't be practical since there is no gain here
//! from processing inputs in a batch.*
//!
//! Suppose we want a public API that allows a user to double a single number, but for some reason
//! its more efficient to double a batch of numbers together. Without batching we could just have
//! a simple function like this:
//!
//! ```rust
//! fn double(x: i32) -> i32 {
//!     x * 2
//! }
//! ```
//!
//! But adding batching behind the scenes is as simple as dropping the `batched_fn` macro
//! in the function body:
//!
//! ```rust
//! # use batched_fn::batched_fn;
//! async fn double(x: i32) -> i32 {
//!     let batched_double = batched_fn! {
//!         |batch: Vec<i32>| -> Vec<i32> {
//!             batch.iter().map(|batch_i| batch_i * 2).collect()
//!         }
//!     };
//!     batched_double(x).await
//! }
//! ```
//!
//! ❗️ *Note that the `double` function now has to be `async`.*
//!
//! We can also adjust the maximum batch size and tune the wait delay.
//!
//! Here we set the `max_batch_size` to 4 and `delay` to 50 milliseconds. This means
//! the batched function will wait at most 50 milliseconds after receiving a single
//! input to fill a batch of 4. If 3 more inputs are not received within 50 milliseconds
//! then the partial batch will be ran as-is.
//!
//! ```rust
//! # use batched_fn::batched_fn;
//! async fn double(x: i32) -> i32 {
//!     let batched_double = batched_fn! {
//!         |batch: Vec<i32>| -> Vec<i32> {
//!             batch.iter().map(|batch_i| batch_i * 2).collect()
//!         },
//!         max_batch_size = 4,
//!         delay = 50,
//!     };
//!     batched_double(x).await
//! }
//! ```

extern crate once_cell;

#[doc(hidden)]
pub use once_cell::sync::Lazy;

use std::marker;
use tokio::sync::mpsc::{self, Sender};
use tokio::time::{self, Duration, Instant};

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

pub trait BatchFn {
    type Input: Batch;
    type Output: Batch;
}

/// A `BatchedFnBuilder` is used to created a `BatchedFn`.
#[doc(hidden)]
pub struct BatchedFnBuilder<T, R, F, BT, BR>
where
    T: 'static + Send + Sync + std::fmt::Debug,
    R: 'static + Send + Sync + std::fmt::Debug,
    F: 'static + Send + Sync + Fn(BT) -> BR,
    BT: Send + Batch<Item = T>,
    BR: Send + Batch<Item = R>,
    <BR as IntoIterator>::IntoIter: Send,
{
    handler: F,
    delay: u32,
    max_batch_size: usize,
    _marker: marker::PhantomData<(T, R, BT, BR)>,
}

impl<T, R, F, BT, BR> BatchedFnBuilder<T, R, F, BT, BR>
where
    T: 'static + Send + Sync + std::fmt::Debug,
    R: 'static + Send + Sync + std::fmt::Debug,
    F: 'static + Send + Sync + Fn(BT) -> BR,
    BT: Send + Batch<Item = T>,
    BR: Send + Batch<Item = R>,
    <BR as IntoIterator>::IntoIter: Send,
{
    /// Get a new `BatchedFnBuilder` with the given `handler`.
    ///
    /// The `handler` is the core of a `BatchedFn` and is responsible for the actual
    /// logic that processes a batch of inputs.
    pub fn new(handler: F) -> Self {
        Self {
            handler,
            delay: 100,
            max_batch_size: 10,
            _marker: marker::PhantomData,
        }
    }

    /// Set the maximum batch size.
    ///
    /// The `BatchedFn` will process at most this many elements in a batch.
    pub fn max_batch_size(mut self, n: usize) -> Self {
        self.max_batch_size = n;
        self
    }

    /// Set the maximum delay in milliseconds to wait to fill a batch.
    ///
    /// Processing of a batch is only delayed when the batch cannot be filled
    /// immediately. This is the trade-off with processing everything in a syncronous
    /// one-at-a-time way.
    pub fn delay(mut self, delay: u32) -> Self {
        self.delay = delay;
        self
    }

    /// Get a `BatchedFn` with the given handler and configuration.
    pub fn build(self) -> BatchedFn<T, R> {
        let (tx, mut rx) = mpsc::channel::<(T, Sender<R>)>(100);
        let handler = self.handler;
        let max_batch_size = self.max_batch_size;
        let delay = self.delay as u128;

        tokio::spawn(async move {
            // Wait for an input.
            while let Some((input, result_tx)) = rx.recv().await {
                let mut batch_input = BT::with_capacity(max_batch_size);
                let mut batch_txs = Vec::with_capacity(max_batch_size);
                batch_input.push(input);
                batch_txs.push(result_tx);

                let mut vacancy = max_batch_size - 1;
                let mut time_left = delay as u64;
                let start = Instant::now();

                // While there is still room in the batch we'll wait at most `delay`
                // milliseconds to try to fill it.
                while vacancy > 0 && time_left > 0 {
                    if let Ok(Some((next_input, next_result_tx))) =
                        time::timeout(Duration::from_millis(time_left), rx.recv()).await
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

                let batch_output = handler(batch_input);
                for (output, mut result_tx) in batch_output.into_iter().zip(batch_txs) {
                    result_tx.send(output).await.unwrap();
                }
            }
        });

        BatchedFn { tx }
    }
}

/// A `BatchedFn` is a wrapper around a `handler` that provides the interface for
/// evaluating a single input as part of a batch of other inputs.
#[doc(hidden)]
#[derive(Clone)]
pub struct BatchedFn<T, R>
where
    T: 'static + Send + Sync + std::fmt::Debug,
    R: 'static + Send + Sync + std::fmt::Debug,
{
    tx: Sender<(T, Sender<R>)>,
}

impl<T, R> BatchedFn<T, R>
where
    T: 'static + Send + Sync + std::fmt::Debug,
    R: 'static + Send + Sync + std::fmt::Debug,
{
    /// Evaluate a single input as part of a batch of other inputs.
    pub async fn evaluate_in_batch(&self, input: T) -> R {
        let (result_tx, mut result_rx) = mpsc::channel::<R>(100);
        self.tx.clone().send((input, result_tx)).await.unwrap();
        result_rx.recv().await.unwrap()
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! __batch_fn_internal {
    (
        |$batch:ident: $batch_input_type:ty| -> $batch_output_type:ty $fn_body:block,
        $( $setting:ident = $value:expr, )*
    ) => {{
        static BATCHED_FN: $crate::Lazy<$crate::BatchedFn<<$batch_input_type as $crate::Batch>::Item, <$batch_output_type as $crate::Batch>::Item>> =
            $crate::Lazy::new(|| {
                let mut builder = $crate::BatchedFnBuilder::new(|$batch: $batch_input_type| -> $batch_output_type { $fn_body });

                $(
                    builder = builder.$setting($value);
                )*

                builder.build()
            });

        |input| { BATCHED_FN.evaluate_in_batch(input) }
    }};
}

/// Macro for creating and calling a batched function.
#[macro_export]
macro_rules! batched_fn {
    (
        |$batch:ident: $batch_input_type:ty| -> $batch_output_type:ty $fn_body:block $(,)?
    ) => {
        $crate::__batch_fn_internal!(
            |$batch: $batch_input_type| -> $batch_output_type $fn_body,
        );
    };
    (
        |$batch:ident: $batch_input_type:ty| -> $batch_output_type:ty $fn_body:block,
        $( $setting:ident = $value:expr ),* $(,)?
    ) => {
        $crate::__batch_fn_internal!(
            |$batch: $batch_input_type| -> $batch_output_type $fn_body,
            $( $setting = $value, )*
        );
    };
}
