# llm-response-cache

An in-memory LRU cache for LLM responses, keyed by a deterministic hash of the
request. It lets agents and applications avoid re-issuing identical model calls
by looking up a previously computed response.

## Overview

Requests are hashed into a stable cache key using **canonical JSON** — objects
are serialized with their keys recursively sorted — so that two semantically
identical requests produce the same key regardless of field ordering. The key
is a hex-encoded **SHA-256** digest (64 characters).

The cache enforces a fixed capacity using a simple **least-recently-used (LRU)**
eviction policy: when the cache is full, the oldest entry is removed to make
room for a new one. Inserting or updating a key marks it as most recently used.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
llm-response-cache = "0.1"
serde_json = "1"
```

## Usage

```rust
use llm_response_cache::ResponseCache;
use serde_json::json;

let mut cache = ResponseCache::new(100); // capacity of 100 entries

let req = json!({
    "model": "claude-opus-4-7",
    "messages": [{"role": "user", "content": "hi"}]
});
let resp = json!({"content": "hello"});

// Insert a response keyed by the request hash; returns the cache key.
let key = cache.insert(&req, resp.clone());

// Later, look it up by the same key.
assert_eq!(cache.get(&key), Some(&resp));
```

You can also compute a key without inserting, or use an explicit key:

```rust
use llm_response_cache::ResponseCache;
use serde_json::json;

// Deterministic key from a request (order-independent).
let key = ResponseCache::compute_key(&json!({"a": 1, "b": 2}));
assert_eq!(key, ResponseCache::compute_key(&json!({"b": 2, "a": 1})));

let mut cache = ResponseCache::new(10);
cache.put("my-key".to_string(), json!({"text": "value"}));
assert!(cache.contains("my-key"));
```

## API

`ResponseCache` provides:

- `new(capacity)` — create a cache with a fixed maximum number of entries.
- `compute_key(request)` — compute the SHA-256 canonical-JSON key for a request.
- `insert(request, response)` — hash the request, store the response, return the key.
- `put(key, value)` — insert or update with an explicit key (LRU-aware).
- `get(key)` — fetch a cached response by key.
- `contains(key)` — check whether a key is cached.
- `remove(key)` — remove and return an entry.
- `clear()` — empty the cache.
- `len()`, `is_empty()`, `capacity()` — size and capacity accessors.

## Tech stack

- **Language:** Rust (edition 2021)
- **Dependencies:** [`serde_json`](https://crates.io/crates/serde_json) for JSON
  handling and [`sha2`](https://crates.io/crates/sha2) for SHA-256 hashing.

## Building and testing

```sh
cargo build
cargo test
```

## License

Licensed under the MIT License.
