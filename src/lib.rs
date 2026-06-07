/*!
`llm-response-cache`: an in-memory LRU cache for LLM responses keyed by a
deterministic hash of the request.

Requests are hashed into a stable cache key using *canonical JSON* (objects are
serialized with their keys recursively sorted), so two semantically identical
requests produce the same key regardless of field ordering. The key is a
hex-encoded SHA-256 digest (64 characters).

The cache enforces a fixed capacity using a least-recently-used (LRU) eviction
policy: when the cache is full, the least recently *used* entry is evicted to
make room. Both inserting/updating a key ([`ResponseCache::put`]) and reading a
key ([`ResponseCache::get`]) mark it as most recently used. Use
[`ResponseCache::peek`] to read an entry without affecting its recency.

```rust
use llm_response_cache::ResponseCache;
use serde_json::json;

let mut cache = ResponseCache::new(100);
let req = json!({"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]});
let resp = json!({"content": "hello"});
let key = cache.insert(&req, resp.clone());
assert_eq!(cache.get(&key), Some(&resp));
```
*/

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};

/// An in-memory LRU cache mapping cache keys to JSON LLM responses.
///
/// Entries are evicted in least-recently-used order once the configured
/// capacity is exceeded. "Use" includes both writes ([`put`](Self::put) /
/// [`insert`](Self::insert)) and reads via [`get`](Self::get). A cache created
/// with capacity `0` stores nothing.
pub struct ResponseCache {
    capacity: usize,
    map: HashMap<String, Value>,
    /// Keys in least-recently-used (front) to most-recently-used (back) order.
    order: VecDeque<String>,
}

impl ResponseCache {
    /// Create a cache that holds at most `capacity` entries.
    ///
    /// A capacity of `0` produces a cache that never stores anything.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Compute a SHA-256 key from a request value (recursive key-sorted JSON).
    pub fn compute_key(request: &Value) -> String {
        let canonical = canonical_json(request);
        let mut h = Sha256::new();
        h.update(canonical.as_bytes());
        format!("{:x}", h.finalize())
    }

    /// Insert a response for a request. Returns the cache key.
    pub fn insert(&mut self, request: &Value, response: Value) -> String {
        let key = Self::compute_key(request);
        self.put(key.clone(), response);
        key
    }

    /// Insert or update a value under an explicit key, marking it most recently
    /// used.
    ///
    /// If the cache is at capacity and `key` is new, the least recently used
    /// entry is evicted first. A cache with capacity `0` discards the value
    /// without storing it.
    pub fn put(&mut self, key: String, value: Value) {
        if self.capacity == 0 {
            return;
        }
        if self.map.contains_key(&key) {
            self.order.retain(|k| k != &key);
        } else if self.map.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.map.remove(&evicted);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }

    /// Get a cached response by key, marking it as most recently used.
    ///
    /// Because reads count as "use" for the LRU policy, this takes `&mut self`.
    /// Use [`peek`](Self::peek) to read without affecting recency.
    pub fn get(&mut self, key: &str) -> Option<&Value> {
        if self.map.contains_key(key) {
            // Move the key to the most-recently-used position.
            self.order.retain(|k| k != key);
            self.order.push_back(key.to_string());
            self.map.get(key)
        } else {
            None
        }
    }

    /// Get a cached response by key without affecting its LRU recency.
    pub fn peek(&self, key: &str) -> Option<&Value> {
        self.map.get(key)
    }

    /// Returns `true` if `key` is currently cached.
    pub fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Maximum number of entries the cache will hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Remove an entry, returning its value if it was present.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        if let Some(v) = self.map.remove(key) {
            self.order.retain(|k| k != key);
            Some(v)
        } else {
            None
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

/// Serialize a JSON value into a canonical string with object keys sorted
/// recursively.
///
/// This produces a deterministic representation so that semantically identical
/// requests hash to the same key regardless of object key ordering. Array order
/// is preserved (arrays are order-significant in JSON).
fn canonical_json(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut pairs: Vec<(&String, &Value)> = map.iter().collect();
            pairs.sort_by_key(|(k, _)| *k);
            let inner: Vec<String> = pairs
                .iter()
                .map(|(k, v)| {
                    // A JSON object key is always a valid string; encode it with
                    // serde_json so quotes/escapes are handled correctly, falling
                    // back to a best-effort form rather than panicking.
                    let key = serde_json::to_string(k).unwrap_or_else(|_| format!("{k:?}"));
                    format!("{}:{}", key, canonical_json(v))
                })
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        Value::Array(arr) => {
            let inner: Vec<String> = arr.iter().map(canonical_json).collect();
            format!("[{}]", inner.join(","))
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn insert_and_get() {
        let mut c = ResponseCache::new(10);
        let req = json!({"model": "claude-opus-4-7", "msg": "hi"});
        let resp = json!({"text": "hello"});
        let key = c.insert(&req, resp.clone());
        assert_eq!(c.get(&key), Some(&resp));
    }

    #[test]
    fn same_request_same_key() {
        let req = json!({"a": 1, "b": 2});
        let k1 = ResponseCache::compute_key(&req);
        let k2 = ResponseCache::compute_key(&req);
        assert_eq!(k1, k2);
    }

    #[test]
    fn different_request_different_key() {
        let k1 = ResponseCache::compute_key(&json!({"a": 1}));
        let k2 = ResponseCache::compute_key(&json!({"a": 2}));
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_order_independent() {
        let k1 = ResponseCache::compute_key(&json!({"a": 1, "b": 2}));
        let k2 = ResponseCache::compute_key(&json!({"b": 2, "a": 1}));
        assert_eq!(k1, k2);
    }

    #[test]
    fn lru_eviction() {
        let mut c = ResponseCache::new(2);
        c.put("k1".into(), json!("v1"));
        c.put("k2".into(), json!("v2"));
        c.put("k3".into(), json!("v3")); // evicts k1
        assert!(!c.contains("k1"));
        assert!(c.contains("k2"));
        assert!(c.contains("k3"));
    }

    #[test]
    fn update_existing_key() {
        let mut c = ResponseCache::new(10);
        c.put("k1".into(), json!("old"));
        c.put("k1".into(), json!("new"));
        assert_eq!(c.get("k1"), Some(&json!("new")));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn contains() {
        let mut c = ResponseCache::new(10);
        c.put("x".into(), json!(1));
        assert!(c.contains("x"));
        assert!(!c.contains("y"));
    }

    #[test]
    fn remove() {
        let mut c = ResponseCache::new(10);
        c.put("k".into(), json!(42));
        assert!(c.remove("k").is_some());
        assert!(!c.contains("k"));
    }

    #[test]
    fn clear() {
        let mut c = ResponseCache::new(10);
        c.put("a".into(), json!(1));
        c.put("b".into(), json!(2));
        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn capacity_getter() {
        let c = ResponseCache::new(50);
        assert_eq!(c.capacity(), 50);
    }

    #[test]
    fn len_tracks_entries() {
        let mut c = ResponseCache::new(10);
        assert_eq!(c.len(), 0);
        c.put("a".into(), json!(1));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn missing_key_returns_none() {
        let mut c = ResponseCache::new(10);
        assert_eq!(c.get("nope"), None);
    }

    #[test]
    fn peek_does_not_change_recency() {
        let mut c = ResponseCache::new(2);
        c.put("k1".into(), json!("v1"));
        c.put("k2".into(), json!("v2"));
        // peek must NOT refresh recency, so k1 stays least-recently-used.
        assert_eq!(c.peek("k1"), Some(&json!("v1")));
        c.put("k3".into(), json!("v3")); // evicts k1 (still oldest)
        assert!(!c.contains("k1"));
        assert!(c.contains("k2"));
        assert!(c.contains("k3"));
    }

    #[test]
    fn get_refreshes_recency() {
        let mut c = ResponseCache::new(2);
        c.put("k1".into(), json!("v1"));
        c.put("k2".into(), json!("v2"));
        // Reading k1 makes it most-recently-used.
        assert_eq!(c.get("k1"), Some(&json!("v1")));
        // Now k2 is least-recently-used and should be evicted, not k1.
        c.put("k3".into(), json!("v3"));
        assert!(c.contains("k1"), "k1 was just read; LRU must keep it");
        assert!(!c.contains("k2"), "k2 was least recently used");
        assert!(c.contains("k3"));
    }

    #[test]
    fn capacity_zero_stores_nothing() {
        let mut c = ResponseCache::new(0);
        c.put("a".into(), json!(1));
        c.put("b".into(), json!(2));
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.get("a"), None);
    }

    #[test]
    fn insert_into_zero_capacity_returns_key_but_stores_nothing() {
        let mut c = ResponseCache::new(0);
        let req = json!({"x": 1});
        let key = c.insert(&req, json!("resp"));
        assert_eq!(key, ResponseCache::compute_key(&req));
        assert!(!c.contains(&key));
    }

    #[test]
    fn order_stays_consistent_after_eviction() {
        // Touch each survivor via get and confirm len never exceeds capacity.
        let mut c = ResponseCache::new(3);
        for i in 0..10 {
            c.put(format!("k{i}"), json!(i));
            assert!(c.len() <= 3);
        }
        // Only the three most-recently-inserted keys remain.
        assert!(c.contains("k9"));
        assert!(c.contains("k8"));
        assert!(c.contains("k7"));
        assert!(!c.contains("k6"));
    }

    #[test]
    fn nested_objects_are_canonicalized() {
        let k1 = ResponseCache::compute_key(&json!({"a": {"x": 1, "y": 2}}));
        let k2 = ResponseCache::compute_key(&json!({"a": {"y": 2, "x": 1}}));
        assert_eq!(k1, k2);
    }

    #[test]
    fn arrays_are_order_sensitive() {
        // Arrays preserve order: different order => different key.
        let k1 = ResponseCache::compute_key(&json!([1, 2, 3]));
        let k2 = ResponseCache::compute_key(&json!([3, 2, 1]));
        assert_ne!(k1, k2);
    }

    #[test]
    fn remove_missing_returns_none() {
        let mut c = ResponseCache::new(10);
        assert_eq!(c.remove("nope"), None);
    }

    #[test]
    fn key_is_hex_string() {
        let key = ResponseCache::compute_key(&json!({"x": 1}));
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(key.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
    }
}
