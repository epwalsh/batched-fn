extern crate once_cell;

#[doc(hidden)]
pub use once_cell::sync::OnceCell;

use std::marker;
use tokio::sync::mpsc::{self, Sender};
use tokio::time::{self, Duration, Instant};

/// The `Batch` trait is essentially an abstraction of `Vec<T>`.
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

    /// This is only used internally within the macro implementation to avoid warnings.
    #[doc(hidden)]
    pub fn touch(&mut self) {}
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
        static BATCHED_FN: $crate::OnceCell<$crate::BatchedFn<<$batch_input_type as $crate::Batch>::Item, <$batch_output_type as $crate::Batch>::Item>> =
            $crate::OnceCell::new();

        async fn call(input: <$batch_input_type as $crate::Batch>::Item) -> <$batch_output_type as $crate::Batch>::Item {
            let batched_fn_wrapper = BATCHED_FN.get_or_init(|| {
                let mut builder = $crate::BatchedFnBuilder::new(|$batch: $batch_input_type| -> $batch_output_type { $fn_body });

                // This is purely so we don't get a warning "builder does not need to be mutable"
                // if there aren't any other settings given.
                builder.touch();

                $(
                    builder = builder.$setting($value);
                )*

                builder.build()
            });

            batched_fn_wrapper.evaluate_in_batch(input).await
        }

        call
    }};
}

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
