//! Deep learning models are usually implemented to make efficient use of a GPU by batching inputs together
//! in "mini-batches". However, applications serving these models often receive requests one-by-one.
//! So using a conventional single or multi-threaded server approach will under-utilize the GPU and lead to latency that increases
//! linearly with the volume of requests.
//!
//! `batched-fn` is a drop-in solution for deep learning webservers that queues individual requests and provides them as a batch
//! to your model. It can be added to any application with minimal refactoring simply by inserting the [`batched_fn`](crate::batched_fn)
//! macro into the function that runs requests through the model.
//!
//! ## Features
//!
//! - 🚀 Easy to use: drop the `batched_fn!` macro into existing code.
//! - 🔥 Lightweight and fast: queue system implemented on top of the blazingly fast [flume crate](https://github.com/zesterer/flume).
//! - 🙌 Easy to tune: simply adjust [`max_delay`](crate::batched_fn#config) and [`max_batch_size`](crate::batched_fn#config).
//! - 🛑 [Back pressure](https://medium.com/@jayphelps/backpressure-explained-the-flow-of-data-through-software-2350b3e77ce7) mechanism included:
//!   just set [`channel_cap`](crate::batched_fn#config) and handle
//!   [`Error::Full`](crate::Error#variant.Full) by returning a 503 from your webserver.
//!
//! ## Examples
//!
//! Suppose you have a model API that look like this:
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
//! Without `batched-fn` a webserver route would need to call `Model::predict` on each
//! individual input, resulting in a bottleneck from under-utilizing the GPU:
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
//! But by dropping the [`batched_fn`](crate::batched_fn) macro into your code you automatically get batched
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
//!             max_delay: 50,
//!         };
//!         context = {
//!             model: Model::load(),
//!         };
//!     };
//!     batch_predict(input).await.unwrap()
//! }
//! ```
//!
//! ❗️ *Note that the `predict_for_http_request` function now has to be `async`.*
//!
//! Here we set the [`max_batch_size`](crate::batched_fn#config) to 16 and [`max_delay`](crate::batched_fn#config)
//! to 50 milliseconds. This means the batched function will wait at most 50 milliseconds after receiving a single
//! input to fill a batch of 16. If 15 more inputs are not received within 50 milliseconds
//! then the partial batch will be ran as-is.
//!
//! ## Tuning max batch size and max delay
//!
//! The optimal batch size and delay will depend on the specifics of your use case, such as how big of a batch you can fit in memory
//! (typically on the order of 8, 16, 32, or 64 for a deep learning model) and how long of a delay you can afford.
//! In general you want to set `max_batch_size` as high as you can, assuming the total processing time for `N` examples is minimized
//! with a batch size of `N`, and keep `max_delay` small relative to the time it takes for your
//! handler function to process a batch.
//!
//! ## Implementation details
//!
//! When the `batched_fn` macro is invoked it spawns a new thread where the
//! [`handler`](crate::batched_fn#handler) will
//! be ran. Within that thread, every object specified in the [`context`](crate::batched_fn#context)
//! is initialized and then passed by reference to the handler each time it is run.
//!
//! The object returned by the macro is just a closure that sends a single input and a callback
//! through an asyncronous channel to the handler thread. When the handler finishes
//! running a batch it invokes the callback corresponding to each input with the corresponding output,
//! which triggers the closure to wake up and return the output.

extern crate flume;
extern crate once_cell;

#[doc(hidden)]
pub use flume::{bounded, unbounded, Sender};
#[doc(hidden)]
pub use once_cell::sync::Lazy;

/// The `Batch` trait is essentially an abstraction of `Vec<T>`. The input and output of a batch
/// [`handler`](crate::batched_fn#handler) must implement `Batch`.
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
pub struct Config {
    pub max_batch_size: usize,
    pub max_delay: u128,
    pub channel_cap: Option<usize>,
    // Used to avoid clippy linting errors within the macro-generated code
    // when updating the fields of this struct.
    pub _phantom: std::marker::PhantomData<bool>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_delay: 50,
            channel_cap: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Error types that can occur while calling a batched function.
#[derive(Debug, Copy, Clone)]
pub enum Error {
    /// Channel is full.
    ///
    /// This can happen if you've set `channel_cap`, and should usually be handled
    /// by returning a 503 error code from your server to signal that the server is too
    /// busy at the moment to handle any more requests.
    Full,

    /// Channel has been disconnected, most likely due to the handler thread crashing.
    Disconnected,
}

/// Created by the [`batched_fn`](crate::batched_fn) macro.
///
/// A `BatchedFn` is a wrapper around a [`handler`](crate::batched_fn#handler)
/// that provides the interface for evaluating a single input as part of a batch of other inputs.
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
    pub fn new(tx: Sender<(T, Sender<R>)>) -> Self {
        Self { tx }
    }

    /// Evaluate a single input as part of a batch of other inputs.
    pub async fn evaluate_in_batch(&self, input: T) -> Result<R, Error> {
        // Can use `unbounded` channel because we already get backpressure from
        // the channel that `self.tx` sends to.
        let (result_tx, result_rx) = unbounded::<R>();
        self.tx.try_send((input, result_tx)).map_err(|e| match e {
            flume::TrySendError::Full(_) => Error::Full,
            flume::TrySendError::Disconnected(_) => Error::Disconnected,
        })?;
        result_rx
            .recv_async()
            .await
            .map_err(|_| Error::Disconnected)
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! __batched_fn_internal {
    (
        handler = |$batch:ident: $batch_input_type:ty $(, $ctx_arg:ident: &$ctx_arg_ty:ty )*| -> $batch_output_type:ty $fn_body:block ;
        config = {
            $( $cfg:ident: $cfg_init:expr ),* $(,)?
        };
        context = {
            $( $ctx:ident: $ctx_init:expr ),* $(,)?
        } $(;)?
    ) => {{
        static BATCHED_FN: $crate::Lazy<
            $crate::BatchedFn<
                <$batch_input_type as $crate::Batch>::Item,
                <$batch_output_type as $crate::Batch>::Item,
            >,
        > = $crate::Lazy::new(|| {
            let config = $crate::Config {
                $( $cfg: $cfg_init, )*
                ..Default::default()
            };

            let (tx, mut rx) = match config.channel_cap {
                None => {
                    $crate::unbounded::<(
                        <$batch_input_type as $crate::Batch>::Item,
                        $crate::Sender<<$batch_output_type as $crate::Batch>::Item>,
                    )>()
                }
                Some(cap) => {
                    $crate::bounded::<(
                        <$batch_input_type as $crate::Batch>::Item,
                        $crate::Sender<<$batch_output_type as $crate::Batch>::Item>,
                    )>(cap)
                }
            };

            std::thread::spawn(move || {
                // Create handler closure.
                let handler = |$batch: $batch_input_type $(, $ctx_arg: &$ctx_arg_ty )*| -> $batch_output_type {
                    $fn_body
                };

                // Set config vars.
                let max_batch_size: usize = config.max_batch_size;
                let max_delay: u128 = config.max_delay;

                // Initialize handler context.
                struct _Context {
                    $( $ctx_arg: $ctx_arg_ty, )*
                }

                let context = _Context {
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
                    let mut time_left = max_delay as u64;
                    let start = std::time::Instant::now();

                    // While there is still room in the batch we'll wait at most `max_delay`
                    // milliseconds to try to fill it.
                    while vacancy > 0 && time_left > 0 {
                        if let Ok((next_input, next_result_tx)) =
                            rx.recv_timeout(std::time::Duration::from_millis(time_left))
                        {
                            batch_input.push(next_input);
                            batch_txs.push(next_result_tx);
                            vacancy -= 1;
                            let elapsed = start.elapsed().as_millis();
                            time_left = if elapsed > max_delay {
                                0
                            } else {
                                (max_delay - elapsed) as u64
                            };
                        } else {
                            break;
                        }
                    }

                    let batch_output = handler(batch_input $(, &context.$ctx_arg )*);
                    for (output, mut result_tx) in batch_output.into_iter().zip(batch_txs) {
                        result_tx.send(output).expect("Channel from calling thread disconnected");
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
/// This macro has 3 parameters: [`handler`](#handler), [`config`](#config), and
/// [`context`](#context). It returns an async function that wraps
/// [`BatchedFn::evaluate_in_batch`](struct.BatchedFn.html#method.evaluate_in_batch).
///
/// # Parameters
///
/// ### `handler`
///
/// The handler must be in the form of a closure declaration that takes a batch
/// and any number of references to objects in the context as input and
/// returns a different type of batch.
///
/// ### `config`
///
/// Within the config you can specify the `max_batch_size`, `max_delay`, and `channel_cap`.
///
/// The batched function will wait at most `max_delay` milliseconds after receiving a single
/// input to fill a batch of size `max_batch_size`. If enough inputs to fill a full batch
/// are not received within `max_delay` milliseconds then the partial batch will be ran as-is.
///
/// The `channel_cap` option allows you to apply back pressure if too many inputs are waiting for
/// the handler thread to accept another batch. By default `channel_cap` is `None`, but if
/// set to `Some(usize)` then
/// [`BatchedFn::evaluate_in_batch`](struct.BatchedFn.html#method.evaluate_in_batch) will
/// return [`Error::Full`](crate::Error#variant.Full) if the channel between the calling thread and the handler thread is at this
/// capacity. You probably want to set this to some multiple of `max_batch_size`.
///
/// ## `context`
///
/// Any additional reference that the handler takes as input must be defined within
/// the context.
///
/// # Examples
///
/// ```rust
/// # #[macro_use] extern crate batched_fn;
/// use batched_fn::{batched_fn, Error};
///
/// async fn double(x: i32) -> Result<i32, Error> {
///     let batched_double = batched_fn! {
///         handler = |batch: Vec<i32>| -> Vec<i32> {
///             batch.into_iter().map(|x| x*2).collect()
///         };
///         config = {
///             max_batch_size: 4,
///             max_delay: 50,
///             channel_cap: Some(20),
///         };
///         context = {};
///     };
///
///     batched_double(x).await
/// }
/// ```
///
/// You can also provide an arbitrary number of additional arguments to the handler by reference.
/// All of the objects have to be initialized in the [`context`](#context):
///
/// ```rust
/// # #[macro_use] extern crate batched_fn;
/// # use batched_fn::{batched_fn, Error};
/// async fn multiply(x: i32) -> Result<i32, Error> {
///     let batched_multiply = batched_fn! {
///         handler = |batch: Vec<i32>, factor: &i32| -> Vec<i32> {
///             batch.into_iter().map(|x| *factor * x ).collect()
///         };
///         config = {
///             max_batch_size: 4,
///             max_delay: 50
///         };
///         context = {
///             factor: 3
///         };
///     };
///
///     batched_multiply(x).await
/// }
/// ```
#[macro_export]
macro_rules! batched_fn {
    (
        handler = |$batch:ident: $batch_input_type:ty $(, $ctx_arg:ident: &$ctx_arg_ty:ty )*| -> $batch_output_type:ty $fn_body:block ;
        config = {
            $( $cfg:ident: $cfg_init:expr ),* $(,)?
        };
        context = {
            $( $ctx:ident: $ctx_init:expr ),* $(,)?
        } $(;)?
    ) => {
        $crate::__batched_fn_internal!(
            handler = |$batch: $batch_input_type $(, $ctx_arg: &$ctx_arg_ty )*| -> $batch_output_type $fn_body ;
            config = {
                $( $cfg: $cfg_init, )*
            };
            context = {
                $( $ctx: $ctx_init, )*
            };
        );
    };
}
