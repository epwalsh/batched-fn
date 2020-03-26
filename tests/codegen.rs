use batched_fn::batched_fn;

fn batch_double(batch: Vec<i32>) -> Vec<i32> {
    let mut out = Vec::with_capacity(batch.len());
    for x in batch {
        out.push(x * 2);
    }
    out
}

fn batch_multiply(batch: Vec<i32>, factor: i32) -> Vec<i32> {
    batch.into_iter().map(|x| x * factor).collect()
}

#[test]
fn test_basic_invocation() {
    let _ = batched_fn! {
        handler = |batch: Vec<i32>| -> Vec<i32> { batch_double(batch) };
        config = {
            max_batch_size: 4,
            delay: 50,
        };
        context = {};
    };
}

#[test]
fn test_with_context() {
    let _ = batched_fn! {
        handler = |batch: Vec<i32>, factor: &i32| -> Vec<i32> { batch_multiply(batch, *factor) };
        config = {
            max_batch_size: 4,
            delay: 50,
        };
        context = {
            factor: 3,
        };
    };
}
