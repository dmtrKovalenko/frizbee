use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::Matcher;
use crate::Match;
use crate::k_merge::k_merge_matches;
use crate::sort::radix_sort_matches;

impl Matcher {
    /// Matches a list of haystacks in parallel on multiple real threads, returning a list of
    /// [`Match`] values. Threads work on 2048 item chunks, which are sorted and merged into a
    /// single sorted `Vec` at the end. The `threads` must be >0.
    ///
    /// This API provides the most performant path when matching on lists.
    pub fn match_list_parallel<S: AsRef<str> + Sync>(
        &mut self,
        haystacks: &[S],
        threads: usize,
    ) -> Vec<Match> {
        Self::guard_against_haystack_overflow(haystacks.len(), 0);
        assert!(threads > 0, "threads must be positive");

        if haystacks.is_empty() || self.needle.is_empty() || threads == 1 {
            return self.match_list(haystacks);
        }

        // Smaller chunks enable better load balancing via stealing
        // but too small increases atomic contention
        let chunk_size = 2048;
        let num_chunks = haystacks.len().div_ceil(chunk_size);
        let next_chunk = AtomicUsize::new(0);

        let matcher = &*self;

        thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut local_matches = Vec::new();
                        let mut matcher = matcher.clone();

                        loop {
                            // Claim next available chunk
                            let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
                            if chunk_idx >= num_chunks {
                                break;
                            }

                            let start = chunk_idx * chunk_size;
                            let end = (start + chunk_size).min(haystacks.len());
                            let haystacks_chunk = &haystacks[start..end];

                            matcher.match_list_into(
                                haystacks_chunk,
                                start as u32,
                                &mut local_matches,
                            );
                        }

                        // Each thread sorts so that we can perform k-way merge
                        if matcher.config.sort {
                            radix_sort_matches(&mut local_matches);
                        }

                        local_matches
                    })
                })
                .collect();

            if matcher.config.sort {
                k_merge_matches(
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap())
                        .collect::<Vec<_>>(),
                )
            } else {
                handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap())
                    .collect()
            }
        })
    }
    /// Matches items in parallel on multiple real threads, resolving each item's haystack bytes
    /// through the `resolve` callback, returning a list of [`Match`] values. Threads work on
    /// 2048 item chunks, which are sorted and merged into a single sorted `Vec` at the end.
    /// The `threads` must be >0.
    ///
    /// See [`Matcher::match_list_resolved_into`] for the resolver contract.
    pub fn match_list_parallel_resolved<T, F, const N: usize>(
        &mut self,
        items: &[T],
        resolve: &F,
        threads: usize,
    ) -> Vec<Match>
    where
        T: Sync,
        F: Fn(&T, &mut [*const u8; N]) -> Option<(usize, u16)> + Sync,
    {
        Self::guard_against_haystack_overflow(items.len(), 0);
        assert!(threads > 0, "threads must be positive");

        if items.is_empty() || self.needle.is_empty() || threads == 1 {
            let mut matches = Vec::new();
            self.match_list_resolved_into(items, 0, resolve, &mut matches);
            if !self.needle.is_empty() && self.config.sort {
                radix_sort_matches(&mut matches);
            }
            return matches;
        }

        // Smaller chunks enable better load balancing via stealing
        // but too small increases atomic contention
        let chunk_size = 2048;
        let num_chunks = items.len().div_ceil(chunk_size);
        let next_chunk = AtomicUsize::new(0);

        let matcher = &*self;

        thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut local_matches = Vec::new();
                        let mut matcher = matcher.clone();

                        loop {
                            // Claim next available chunk
                            let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
                            if chunk_idx >= num_chunks {
                                break;
                            }

                            let start = chunk_idx * chunk_size;
                            let end = (start + chunk_size).min(items.len());

                            matcher.match_list_resolved_into(
                                &items[start..end],
                                start as u32,
                                resolve,
                                &mut local_matches,
                            );
                        }

                        // Each thread sorts so that we can perform k-way merge
                        if matcher.config.sort {
                            radix_sort_matches(&mut local_matches);
                        }

                        local_matches
                    })
                })
                .collect();

            if matcher.config.sort {
                k_merge_matches(
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap())
                        .collect::<Vec<_>>(),
                )
            } else {
                handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap())
                    .collect()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::r#const::SIMD_CHUNK_BYTES;
    use crate::{Config, Matcher, match_list, match_list_parallel, match_list_parallel_resolved};

    fn thread_counts() -> &'static [usize] {
        if cfg!(miri) {
            &[1, 2, 8]
        } else {
            &[1, 2, 3, 4, 5, 6, 7, 8]
        }
    }

    #[test]
    fn sorted_matches_sequential_across_chunk_boundaries() {
        let mut haystacks = (0..4101)
            .map(|index| format!("nomatch-{index}"))
            .collect::<Vec<_>>();
        for (index, value) in [
            (0, "abc"),
            (2047, "xabc"),
            (2048, "abxc"),
            (2049, "alpha/beta/abc"),
            (4095, "ABC"),
            (4096, "a_b_c"),
            (4100, "zabc"),
        ] {
            haystacks[index] = value.to_string();
        }

        let config = Config {
            sort: true,
            ..Config::default()
        };
        let sequential = match_list("abc", &haystacks, &config);
        assert!(sequential.is_sorted());

        for &threads in thread_counts() {
            let parallel = match_list_parallel("abc", &haystacks, &config, threads);
            assert_eq!(&parallel, &sequential, "threads={threads}");
            assert!(parallel.is_sorted(), "threads={threads}");
        }
    }

    #[test]
    #[should_panic(expected = "threads must be positive")]
    fn zero_threads_panics() {
        let _ = match_list_parallel("a", &["a"], &Config::default(), 0);
    }

    /// Chunked haystack test item. Raw pointers are not `Sync`, so an arena-based caller
    /// wraps them in a type that guarantees the backing memory is immutable and alive.
    #[derive(Clone)]
    struct ChunkItem {
        ptrs: Vec<*const u8>,
        chunk_count: usize,
        byte_len: u16,
    }
    unsafe impl Sync for ChunkItem {}

    /// Splits a string into leaked 16-byte zero-padded chunks, as an arena-based caller would
    /// provide them to the resolved matching APIs.
    fn string_to_chunks(s: &str) -> ChunkItem {
        let bytes = s.as_bytes();
        let n_chunks = if bytes.is_empty() {
            0
        } else {
            bytes.len().div_ceil(SIMD_CHUNK_BYTES)
        };
        let mut arena = vec![[0u8; SIMD_CHUNK_BYTES]; n_chunks];
        for (i, chunk) in arena.iter_mut().enumerate() {
            let start = i * SIMD_CHUNK_BYTES;
            let take = SIMD_CHUNK_BYTES.min(bytes.len() - start);
            chunk[..take].copy_from_slice(&bytes[start..start + take]);
        }
        let ptrs: Vec<*const u8> = arena.iter().map(|c| c.as_ptr()).collect();
        std::mem::forget(arena);
        ChunkItem {
            ptrs,
            chunk_count: n_chunks,
            byte_len: bytes.len() as u16,
        }
    }

    fn resolve_chunks<const N: usize>(
        item: &ChunkItem,
        ptrs_buf: &mut [*const u8; N],
    ) -> Option<(usize, u16)> {
        ptrs_buf[..item.ptrs.len()].copy_from_slice(&item.ptrs);
        Some((item.chunk_count, item.byte_len))
    }

    /// Resolved matching must produce the same set of matched indices and
    /// scores as contiguous matching for arbitrary needle/haystack pairs.
    #[test]
    fn resolved_matches_contiguous_parity() {
        use proptest::prelude::*;
        use proptest::test_runner::{Config as PropConfig, TestRunner};

        let mut runner = TestRunner::new(PropConfig {
            cases: if cfg!(miri) { 16 } else { 2000 },
            ..PropConfig::default()
        });

        let strategy = (
            "[a-z]{2,12}",                                        // needle
            proptest::collection::vec("[a-z/_\\.]{5,80}", 1..30), // haystacks
            (0u16..=8u16),                                        // max_typos
        );

        runner
            .run(&strategy, |(needle, haystacks, max_typos)| {
                let config = Config {
                    max_typos: Some(max_typos),
                    sort: false,
                    ..Config::default()
                };

                // Contiguous path
                let haystack_refs: Vec<&str> = haystacks.iter().map(String::as_str).collect();
                let contiguous = match_list(&needle, &haystack_refs, &config);

                // Build chunk data for each haystack
                let chunk_data: Vec<ChunkItem> =
                    haystacks.iter().map(|s| string_to_chunks(s)).collect();

                // Resolved path
                let mut matcher = Matcher::new(&needle, &config);
                let mut resolved = Vec::new();
                matcher.match_list_resolved_into(
                    &chunk_data,
                    0,
                    &resolve_chunks::<8>,
                    &mut resolved,
                );

                prop_assert_eq!(
                    &contiguous,
                    &resolved,
                    "needle={:?} max_typos={}",
                    needle,
                    max_typos,
                );

                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn resolved_parallel_matches_sequential_across_chunk_boundaries() {
        let mut haystacks = (0..4101)
            .map(|index| format!("nomatch-{index}"))
            .collect::<Vec<_>>();
        for (index, value) in [
            (0, "abc"),
            (2047, "xabc"),
            (2048, "abxc"),
            (2049, "alpha/beta/abc"),
            (4095, "ABC"),
            (4096, "a_b_c"),
            (4100, "zabc"),
        ] {
            haystacks[index] = value.to_string();
        }

        let config = Config {
            sort: true,
            ..Config::default()
        };
        let sequential = match_list("abc", &haystacks, &config);
        assert!(sequential.is_sorted());

        let chunk_data: Vec<ChunkItem> = haystacks.iter().map(|s| string_to_chunks(s)).collect();

        for &threads in thread_counts() {
            let parallel = match_list_parallel_resolved(
                "abc",
                &chunk_data,
                &resolve_chunks::<2>,
                &config,
                threads,
            );
            assert_eq!(&parallel, &sequential, "threads={threads}");
            assert!(parallel.is_sorted(), "threads={threads}");
        }
    }

    #[test]
    fn resolved_skips_none_items_and_empty_needle_reports_present_items() {
        let present = string_to_chunks("hello_world");
        let items = [Some(present.clone()), None, Some(present)];
        let resolve =
            |item: &Option<ChunkItem>, ptrs_buf: &mut [*const u8; 4]| -> Option<(usize, u16)> {
                item.as_ref()
                    .and_then(|item| resolve_chunks(item, ptrs_buf))
            };

        let config = Config {
            sort: false,
            ..Config::default()
        };
        let matches = match_list_parallel_resolved("hw", &items, &resolve, &config, 2);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 2]
        );

        // Empty needle reports every resolvable item
        let matches = match_list_parallel_resolved("", &items, &resolve, &config, 2);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 2]
        );
    }
}
