use warp::Filter;

fn sum_vals(a: i64, b: i64) -> String {
    let sum = a + b;
    println!("{sum}");
    return format!("{sum}")
}

#[tokio::main]
async fn main() {
    // let hello = warp::path!("hello" / i64 / i64).map(sum_vals);
    // warp::serve(hello).run(([127, 0, 0, 1], 3030)).await;

    
}
