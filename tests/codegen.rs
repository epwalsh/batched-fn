use batched_fn::batched_fn;

#[tokio::test]
async fn test_basic_invocation() {
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> {
            println!("Processing batch {:?}", batch);
            let mut out = Vec::with_capacity(batch.len());
            for x in batch {
                out.push(x * 2);
            }
            out
        }
    };
}

#[tokio::test]
async fn test_basic_invocation_trailing_comma() {
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> {
            println!("Processing batch {:?}", batch);
            let mut out = Vec::with_capacity(batch.len());
            for x in batch {
                out.push(x * 2);
            }
            out
        },
    };
}

#[tokio::test]
async fn test_with_options() {
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> {
            println!("Processing batch {:?}", batch);
            let mut out = Vec::with_capacity(batch.len());
            for x in batch {
                out.push(x * 2);
            }
            out
        },
        max_batch_size = 4,
        delay = 50
    };
}

#[tokio::test]
async fn test_with_options_and_trailing_comma() {
    let _ = batched_fn! {
        |batch: Vec<i32>| -> Vec<i32> {
            println!("Processing batch {:?}", batch);
            let mut out = Vec::with_capacity(batch.len());
            for x in batch {
                out.push(x * 2);
            }
            out
        },
        max_batch_size = 4,
        delay = 50,
    };
}
