// This example shows how you could use `batched_fn!` with a deep learing model
// to get batched inference.
//
// Here our `Model` class just takes a vector of i32s and doubles them in order
// to simulate some computation.

// For lazily loading a static reference to a model instance.
use batched_fn::batched_fn;
use once_cell::sync::Lazy;
use tokio::time::{self, Duration};

// `Batch` could be anything that implements the `batched_fn::Batch` trait.
// We'll just use a Vec<T> type since the `Batch` trait has a blanket implementation
// for Vecs.
type Batch<T> = Vec<T>;

type Input = i32;

type Output = i32;

struct Model {}

// This is our stupid model that just doubles the inputs.
impl Model {
    fn predict(&self, batch: Batch<Input>) -> Batch<Output> {
        batch.iter().map(|input| input * 2).collect()
    }

    fn load() -> Self {
        Self {}
    }
}

// Load a model instance.
static MODEL: Lazy<Model> = Lazy::new(Model::load);

async fn predict_for_single_input(input: Input) -> Output {
    let batch_predict = batched_fn! {
        |batch: Batch<Input>| -> Batch<Output> {
            let output = MODEL.predict(batch.clone());
            println!("Processed batch {:?} -> {:?}", batch, output);
            output
        },
        max_batch_size = 4,
        delay = 50,
    };
    batch_predict(input).await
}

#[tokio::main]
async fn main() {
    let mut handles = vec![];

    handles.push(tokio::spawn(async move {
        predict_for_single_input(0).await;
    }));

    // Pause for a moment before firing off some more tasks.
    time::delay_for(Duration::from_millis(10)).await;

    for i in 1..10 {
        handles.push(tokio::spawn(async move {
            predict_for_single_input(i).await;
        }));
    }

    // Wait for spawn tasks to finish.
    for join_handle in handles {
        join_handle.await.unwrap();
    }
}
