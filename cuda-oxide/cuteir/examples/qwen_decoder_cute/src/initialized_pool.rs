/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Bounded reuse of initialized allocations, with exclusive checked-out ownership.

use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, Weak},
};

type Key = (usize, usize);
struct Idle<T> {
    values: HashMap<Key, Vec<T>>,
    bytes: usize,
}
struct State<T> {
    idle: Mutex<Idle<T>>,
    budget: usize,
    single: usize,
    per_size: usize,
    complete: fn(&T) -> bool,
}
impl<T> State<T> {
    fn can_retain(&self, idle: &Idle<T>, key: Key) -> bool {
        let bytes = key.1;
        bytes != 0
            && bytes <= self.single
            && bytes <= self.budget.saturating_sub(idle.bytes)
            && idle.values.get(&key).map_or(0, Vec::len) < self.per_size
    }
}

/// A pool never owns a checked-out allocation. Its weak leases may outlive it.
/// `complete` must establish that no external asynchronous operation still uses
/// the value before it can become eligible for exclusive reuse.
pub(crate) struct Pool<T> {
    state: Arc<State<T>>,
}
impl<T> Clone for Pool<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}
pub(crate) struct Lease<T> {
    value: Option<T>,
    pool: Weak<State<T>>,
    key: Key,
}

impl<T> Pool<T> {
    pub(crate) fn new(
        bytes: usize,
        single: usize,
        per_size: usize,
        complete: fn(&T) -> bool,
    ) -> Self {
        Self {
            state: Arc::new(State {
                idle: Mutex::new(Idle {
                    values: HashMap::new(),
                    bytes: 0,
                }),
                budget: bytes,
                single,
                per_size,
                complete,
            }),
        }
    }
    pub(crate) fn acquire<E>(
        &self,
        key: usize,
        bytes: usize,
        create: impl FnOnce() -> Result<T, E>,
    ) -> Result<Lease<T>, E> {
        let key = (key, bytes);
        let reused = self.state.idle.lock().ok().and_then(|mut idle| {
            let bucket = idle.values.get_mut(&key)?;
            let value = bucket.pop()?;
            if bucket.is_empty() {
                idle.values.remove(&key);
            }
            idle.bytes -= bytes;
            Some(value)
        });
        let value = match reused {
            Some(value) => value,
            None => create()?,
        };
        Ok(Lease {
            value: Some(value),
            pool: Arc::downgrade(&self.state),
            key,
        })
    }
}
impl<T> Lease<T> {
    pub(crate) fn owned(value: T) -> Self {
        Self {
            value: Some(value),
            pool: Weak::new(),
            key: (0, 0),
        }
    }
}
impl<T> Deref for Lease<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().expect("live allocation lease")
    }
}
impl<T> DerefMut for Lease<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("live allocation lease")
    }
}
impl<T> Drop for Lease<T> {
    fn drop(&mut self) {
        let Some(pool) = self.pool.upgrade() else {
            return;
        };
        let eligible = pool
            .idle
            .lock()
            .ok()
            .is_some_and(|idle| pool.can_retain(&idle, self.key));
        if !eligible || !(pool.complete)(self.value.as_ref().expect("live allocation lease")) {
            return;
        }
        // Completion runs outside the mutex. Recheck the budget after another
        // thread may have returned an allocation while the fence was pending.
        if let Ok(mut idle) = pool.idle.lock()
            && pool.can_retain(&idle, self.key)
        {
            idle.values
                .entry(self.key)
                .or_default()
                .push(self.value.take().expect("live allocation lease"));
            idle.bytes += self.key.1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct Allocation {
        values: Vec<u8>,
        can_complete: bool,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Allocation {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn make(id: u8, drops: &Arc<AtomicUsize>) -> Allocation {
        Allocation {
            values: vec![id; 4],
            can_complete: true,
            drops: drops.clone(),
        }
    }
    fn complete(value: &Allocation) -> bool {
        value.can_complete
    }
    fn acquire(
        pool: &Pool<Allocation>,
        key: usize,
        bytes: usize,
        id: u8,
        drops: &Arc<AtomicUsize>,
    ) -> Lease<Allocation> {
        pool.acquire(key, bytes, || Ok::<_, ()>(make(id, drops)))
            .unwrap()
    }

    #[test]
    fn retains_initialized_values_without_reusing_a_live_lease() {
        let drops = Arc::new(AtomicUsize::new(0));
        let pool = Pool::new(16, 8, 2, complete);
        let mut first = acquire(&pool, 4, 4, 1, &drops);
        first.values[0] = 7;
        let second = acquire(&pool, 4, 4, 2, &drops);
        assert_eq!(second.values, [2; 4]);
        drop(first);
        let third = acquire(&pool, 4, 4, 3, &drops);
        assert_eq!(third.values, [7, 1, 1, 1]);
        assert_eq!(second.values, [2; 4]);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn exact_size_and_retention_limits_do_not_alias_or_grow_unbounded() {
        let drops = Arc::new(AtomicUsize::new(0));
        let pool = Pool::new(8, 4, 1, complete);
        let one = acquire(&pool, 1, 4, 1, &drops);
        let duplicate = acquire(&pool, 1, 4, 2, &drops);
        drop(one);
        drop(duplicate); // only one idle allocation per exact size
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(acquire(&pool, 2, 4, 3, &drops));
        drop(acquire(&pool, 3, 4, 4, &drops)); // aggregate budget exhausted
        drop(acquire(&pool, 4, 8, 5, &drops)); // single-allocation limit
        assert_eq!(drops.load(Ordering::SeqCst), 3);
        let reused = acquire(&pool, 1, 4, 6, &drops);
        assert_eq!(reused.values, [1; 4]);
        let different_extent = acquire(&pool, 1, 2, 7, &drops);
        assert_eq!(different_extent.values, [7; 4]);
    }

    #[test]
    fn failed_completion_discards_instead_of_reusing() {
        let drops = Arc::new(AtomicUsize::new(0));
        let pool = Pool::new(16, 8, 2, complete);
        let mut value = acquire(&pool, 4, 4, 1, &drops);
        value.can_complete = false;
        drop(value);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let replacement = acquire(&pool, 4, 4, 2, &drops);
        assert_eq!(replacement.values, [2; 4]);
        drop(replacement);
        let reused = acquire(&pool, 4, 4, 3, &drops);
        assert_eq!(reused.values, [2; 4]);
    }

    #[test]
    fn owner_drop_keeps_checked_out_values_valid_and_releases_idle_values() {
        let drops = Arc::new(AtomicUsize::new(0));
        let pool = Pool::new(16, 8, 2, complete);
        let live = acquire(&pool, 4, 4, 1, &drops);
        drop(acquire(&pool, 4, 4, 2, &drops));
        drop(pool);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(live.values, [1; 4]);
        drop(live);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn failed_creation_does_not_consume_idle_budget() {
        let drops = Arc::new(AtomicUsize::new(0));
        let pool = Pool::new(4, 4, 1, complete);
        assert!(
            pool.acquire(4, 4, || Err::<Allocation, _>("allocation failed"))
                .is_err()
        );
        drop(acquire(&pool, 4, 4, 1, &drops));
        assert_eq!(acquire(&pool, 4, 4, 2, &drops).values, [1; 4]);
    }
}
