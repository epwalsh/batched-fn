use batched_fn::batched_fn;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use tokio::time::{self, Duration};

type TestResult = (Vec<i32>, Vec<i32>);
static TEST_RESULTS: Lazy<Mutex<Vec<TestResult>>> = Lazy::new(|| Mutex::new(vec![]));

async fn double(x: i32) -> i32 {
    let batched = batched_fn! {
        handler = |batch: Vec<i32>| -> Vec<i32> {
            let mut out = Vec::with_capacity(batch.len());
            for x in &batch {
                out.push(x * 2);
            }
            TEST_RESULTS.lock().unwrap().push((batch, out.clone()));
            out
        };
        config = {
            max_batch_size: 4,
            max_delay: 50,
        };
        context = {};
    };

    batched(x).await.unwrap()
}

#[tokio::test]
async fn test_runtime() {
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

    // Now check that the batches are correct.
    let results = TEST_RESULTS.lock().unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0.len(), 4);
    assert_eq!(results[1].0.len(), 4);
    assert_eq!(results[2].0.len(), 2);

    assert_eq!(results[0].0[0], 0);
}
