use ndarray::prelude::*;
use onnxruntime::{environment::Environment, ndarray, GraphOptimizationLevel, LoggingLevel};
use std::array;

fn sum_vals(a: i64, b: i64) -> String {
    let sum = a + b;
    println!("{sum}");
    return format!("{sum}");
}

#[tokio::main]
async fn main() {
    // let hello = warp::path!("hello" / i64 / i64).map(sum_vals);
    // warp::serve(hello).run(([127, 0, 0, 1], 3030)).await;

    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()
        .unwrap();

    let session = environment
        .new_session_builder()
        .unwrap() // alternative?
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_number_threads(1)
        .unwrap()
        .with_model_from_file("../../artifacts/model.onnx")
        .unwrap();

    // TBD
    let input = vec![
        array![1.0],
        array![String::from("Female")],
        array![String::from("No")],
        array![String::from("Sun")],
        array![String::from("Dinner")],
        array![2],
    ];
}
