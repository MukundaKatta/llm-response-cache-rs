/*!
llm-response-cache: LRU cache for LLM responses keyed by request hash.

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
use std::collections::{HashMap, VecDeque};
use sha2::{Sha256, Digest};

/// An LRU cache for LLM responses.
pub struct ResponseCache {
    capacity: usize,
    map: HashMap<String, Value>,
    order: VecDeque<String>,
}

impl ResponseCache {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, map: HashMap::new(), order: VecDeque::new() }
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

    /// Insert with an explicit key.
    pub fn put(&mut self, key: String, value: Value) {
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

    /// Get a cached response by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.map.get(key)
    }

    /// True if key is cached.
    pub fn contains(&self, key: &str) -> bool { self.map.contains_key(key) }

    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }
    pub fn capacity(&self) -> usize { self.capacity }

    /// Remove an entry.
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

/// Recursive key-sorted JSON serialization.
fn canonical_json(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut pairs: Vec<(&String, &Value)> = map.iter().collect();
            pairs.sort_by_key(|(k, _)| *k);
            let inner: Vec<String> = pairs.iter().map(|(k, v)| {
                format!("{}:{}", serde_json::to_string(k).unwrap(), canonical_json(v))
            }).collect();
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
        let c = ResponseCache::new(10);
        assert_eq!(c.get("nope"), None);
    }

    #[test]
    fn key_is_hex_string() {
        let key = ResponseCache::compute_key(&json!({"x": 1}));
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(key.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
    }
}
