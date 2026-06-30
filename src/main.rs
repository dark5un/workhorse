#[tokio::main]
async fn main() -> anyhow::Result<()> {
    workhorse::cli::run().await
}
