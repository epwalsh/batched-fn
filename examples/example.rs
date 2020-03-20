use batched_fn::batched_fn;
use tokio::time::{self, Duration};

async fn double(x: i32) -> i32 {
    let batched = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> {
            let mut out = Vec::with_capacity(batch.len());
            for x in &batch {
                out.push(x * 2);
            }
            println!("Processed batch {:?} -> {:?}", batch, out);
            out
        },
        max_batch_size = 4,
        delay = 50,
    };
    batched(x).await
}

#[tokio::main]
async fn main() {
    let mut handles = vec![];

    handles.push(tokio::spawn(async move {
        double(0).await;
    }));

    // Pause for a moment before firing off some more tasks.
    time::delay_for(Duration::from_millis(10)).await;

    for i in 1..10 {
        handles.push(tokio::spawn(async move {
            double(i).await;
        }));
    }

    // Wait for spawn tasks to finish.
    for join_handle in handles {
        join_handle.await.unwrap();
    }
}
