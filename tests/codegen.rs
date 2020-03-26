use batched_fn::batched_fn;

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
        |batch: Vec<i32>| -> Vec<i32> { batch_double(batch) },
        delay = 50,
        max_batch_size = 4
    };
}

#[tokio::test]
async fn test_basic_invocation_trailing_comma() {
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> { batch_double(batch) },
        delay = 50,
        max_batch_size = 4,
    };
}

#[tokio::test]
async fn test_with_options_diff_order() {
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> { batch_double(batch) },
        max_batch_size = 4,
        delay = 50
    };
}

#[tokio::test]
async fn test_with_options_diff_order_and_trailing_comma() {
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> { batch_double(batch) },
        max_batch_size = 4,
        delay = 50,
    };
}

#[derive(Default)]
struct Context {}

#[tokio::test]
async fn test_basic_invocation_with_ctx() {
    let _ = batched_fn! {
        |batch: Vec<i32>, _ctx: &Context| -> Vec<i32> { batch_double(batch) },
        delay = 50,
        max_batch_size = 4
    };
}

#[tokio::test]
async fn test_basic_invocation_trailing_comma_with_ctx() {
    let _ = batched_fn! {
        |batch: Vec<i32>, _ctx: &Context| -> Vec<i32> { batch_double(batch) },
        delay = 50,
        max_batch_size = 4,
    };
}

#[tokio::test]
async fn test_with_options_diff_order_with_ctx() {
    let _ = batched_fn! {
        |batch: Vec<i32>, _ctx: &Context| -> Vec<i32> { batch_double(batch) },
        max_batch_size = 4,
        delay = 50
    };
}

#[tokio::test]
async fn test_with_options_diff_order_and_trailing_comma_with_ctx() {
    let _ = batched_fn! {
        |batch: Vec<i32>, _ctx: &Context| -> Vec<i32> { batch_double(batch) },
        max_batch_size = 4,
        delay = 50,
    };
}
