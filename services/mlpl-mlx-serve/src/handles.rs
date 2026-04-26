//! Peer-side tensor handle store. Saga R1 step 002.
//!
//! Tensors returned by the eval-on-device endpoint
//! live on the peer's heap; the orchestrator only
//! holds an opaque handle. The handle store maps
//! that handle to the actual `DenseArray`. Cleanup
//! happens via explicit
//! `POST /v1/sessions/{id}/release-handle/{h}` or
//! when the orchestrator's session is dropped.

use std::collections::HashMap;
use std::sync::Arc;

use mlpl_array::DenseArray;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Shared handle store. `Arc<RwLock<...>>` so axum
/// handlers can clone cheaply while serializing
/// writes.
pub type HandleStore = Arc<RwLock<HashMap<Uuid, DenseArray>>>;

/// Construct a fresh empty handle store.
#[must_use]
pub fn new_store() -> HandleStore {
    Arc::new(RwLock::new(HashMap::new()))
}

/// Insert a tensor under a fresh uuid handle and
/// return the handle. Caller surfaces the handle
/// back to the orchestrator as the result of the
/// eval-on-device call.
pub async fn insert(store: &HandleStore, arr: DenseArray) -> Uuid {
    let id = Uuid::new_v4();
    store.write().await.insert(id, arr);
    id
}

/// Look up a tensor by handle. Returns `None` if
/// the handle is unknown (already released, never
/// existed, expired across a server restart).
pub async fn get(store: &HandleStore, id: &Uuid) -> Option<DenseArray> {
    store.read().await.get(id).cloned()
}

/// Remove a handle, returning the tensor if it
/// existed. Cleanup hook for explicit release +
/// session drop.
pub async fn release(store: &HandleStore, id: &Uuid) -> Option<DenseArray> {
    store.write().await.remove(id)
}
