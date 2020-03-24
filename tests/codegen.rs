use batched_fn::batched_fn;
use once_cell::sync::Lazy;

fn batch_double(batch: Vec<i32>) -> Vec<i32> {
    println!("Processing batch {:?}", batch);
    let mut out = Vec::with_capacity(batch.len());
    for x in batch {
        out.push(x * 2);
    }
    out
}

#[tokio::test]
async fn test_basic_invocation() {
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> { batch_double(batch) }
    };
}

#[tokio::test]
async fn test_basic_invocation_trailing_comma() {
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> { batch_double(batch) },
    };
}

#[tokio::test]
async fn test_with_options() {
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> { batch_double(batch) },
        max_batch_size = 4,
        delay = 50
    };
}

#[tokio::test]
async fn test_with_options_and_trailing_comma() {
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> { batch_double(batch) },
        max_batch_size = 4,
        delay = 50,
    };
}

#[tokio::test]
async fn test_handler_with_static_lazy_data() {
    static FACTOR: Lazy<i32> = Lazy::new(|| 2);
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> {
            batch.iter().map(|batch_i| batch_i * (*FACTOR)).collect()
        },
    };
}
