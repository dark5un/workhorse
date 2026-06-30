FROM rust:1.95-slim AS builder

WORKDIR /app

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release || true \
    && rm -rf src

# Copy source and build
COPY . .
RUN cargo build --release --bin myharness --bin mock_mcp_server

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/myharness /usr/local/bin/myharness
COPY --from=builder /app/target/release/mock_mcp_server /usr/local/bin/mock_mcp_server
COPY --from=builder /app/config /app/config

WORKDIR /app

ENV HARNESS_SESSION__PATH=/data/sessions.db

VOLUME ["/data"]

ENTRYPOINT ["myharness"]
