// This example shows how you could use `batched_fn!` with a deep learing model
// to get batched inference.
//
// Here our `Model` class just takes a vector of i32s and doubles them in order
// to simulate some computation.

// For lazily loading a static reference to a model instance.
use batched_fn::batched_fn;
use tokio::time::{self, Duration};

// `Batch` could be anything that implements the `batched_fn::Batch` trait.
// We'll just use a Vec<T> type since the `Batch` trait has a blanket implementation
// for Vecs.
type Batch<T> = Vec<T>;

type Input = i32;

type Output = i32;

struct Model {}

// This is our silly model that just doubles the inputs.
impl Model {
    fn predict(&self, batch: Batch<Input>) -> Batch<Output> {
        batch.iter().map(|input| input * 2).collect()
    }

    fn load() -> Self {
        Self {}
    }
}

async fn predict_for_single_input(input: Input) -> Output {
    let batched_predictor = batched_fn! {
        handler = |batch: Batch<Input>, model: &Model| -> Batch<Output> {
            let output = model.predict(batch.clone());
            println!("Processed batch {:?} -> {:?}", batch, output);
            output
        };
        config = {
            max_batch_size: 4,
            max_delay: 50,
        };
        context = {
            model: Model::load(),
        };
    };
    batched_predictor.evaluate_in_batch(input).await.unwrap()
}

#[tokio::main]
async fn main() {
    let mut handles = vec![tokio::spawn(async move {
        let o = predict_for_single_input(0).await;
        println!("0 -> {}", o);
    })];

    // Pause for a moment before firing off some more tasks.
    time::sleep(Duration::from_millis(10)).await;

    for i in 1..10 {
        handles.push(tokio::spawn(async move {
            let o = predict_for_single_input(i).await;
            println!("{} -> {}", i, o);
        }));
    }

    // Wait for spawn tasks to finish.
    for join_handle in handles {
        join_handle.await.unwrap();
    }
}
