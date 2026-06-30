#[tokio::main]
async fn main() -> anyhow::Result<()> {
    myharness::cli::run().await
}
