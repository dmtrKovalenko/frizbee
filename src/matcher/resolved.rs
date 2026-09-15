//! Resolver-based matching for haystacks whose bytes live in non-contiguous
//! storage (e.g. a chunked arena). Each item is resolved to a list of
//! [`SIMD_CHUNK_BYTES`]-wide chunk pointers, gathered into a contiguous scratch
//! buffer in small batches, and matched through the regular
//! [`Matcher::match_list`] pipeline. This keeps the resolved path on the same
//! backends, matching modes and multi-pattern logic as contiguous matching, at
//! the cost of one memcpy per candidate.

use super::Matcher;
use super::multi::CompiledPatterns;
use crate::Match;
use crate::r#const::SIMD_CHUNK_BYTES;
use crate::sort::radix_sort_matches;
use alloc::vec::Vec;
use core::ops::Range;

/// Items gathered per batch. Bounds the scratch buffer to
/// `GATHER_BATCH * max_haystack_len` bytes so it stays cache-resident while the
/// batch is matched.
const GATHER_BATCH: usize = 64;

/// Reusable buffers for the resolver-based matching loop. One per thread in the
/// parallel API, reused across every chunk it processes, so the hot loop never
/// regrows its vectors.
#[derive(Debug, Default)]
pub(super) struct ResolvedScratch {
    /// Gathered haystack bytes for the current batch (chunk-granular writes, so it
    /// is always reserved to a multiple of [`SIMD_CHUNK_BYTES`])
    bytes: Vec<u8>,
    /// (item index, byte offset into `bytes`, byte len) per gathered item
    spans: Vec<(u32, u32, u32)>,
    hits: Vec<Match>,
}

impl Matcher {
    /// Matches `len` items whose haystack bytes are resolved by index through
    /// a caller-provided callback, returning a list of [`Match`] values
    /// ordered by the configured [`crate::SortStrategy`]. This is the
    /// primitive behind the slice-based resolver APIs: it needs no contiguous
    /// slice of items and is instantiated once per resolver closure.
    ///
    /// See [`Matcher::match_range_resolved_into`] for the resolver contract.
    pub fn match_range_resolved<F, const N: usize>(&mut self, len: usize, resolve: &F) -> Vec<Match>
    where
        F: Fn(u32, &mut [*const u8; N]) -> Option<(usize, u16)>,
    {
        Self::guard_against_haystack_overflow(len, 0);
        let mut matches = Vec::new();
        self.match_range_resolved_into(0..len as u32, resolve, &mut matches);
        if self.config.sort.is_reversed() {
            matches.reverse();
        }
        if !self.patterns.is_empty() && self.config.sort.is_by_score() {
            radix_sort_matches(&mut matches);
        }
        matches
    }

    /// Matches the items with indices in `range`, resolving each one's haystack
    /// bytes through `resolve`, and appends the results to `matches` in index
    /// order (unsorted). Match indices are the indices passed to `resolve`.
    ///
    /// For each index, `resolve` is called with a stack buffer. It should fill
    /// the buffer with pointers to [`crate::SIMD_CHUNK_BYTES`]-wide chunks of
    /// the haystack and return `Some((chunk_count, byte_len))`, or `None` to
    /// skip the item (e.g. deleted files). `chunk_count` must equal
    /// `byte_len.div_ceil(SIMD_CHUNK_BYTES)`.
    ///
    /// `N` is the chunk pointer capacity and must cover the longest haystack:
    /// `max_haystack_bytes.div_ceil(SIMD_CHUNK_BYTES)`.
    ///
    /// # Pointer contract
    /// Every returned chunk pointer must be readable for the full
    /// [`crate::SIMD_CHUNK_BYTES`] bytes (i.e. the last chunk is padded, as in
    /// a chunk arena), and the first `byte_len` gathered bytes must form valid
    /// UTF-8 (i.e. the chunks were produced by splitting a `str`). Violating
    /// this results in undefined behavior.
    pub fn match_range_resolved_into<F, const N: usize>(
        &mut self,
        range: Range<u32>,
        resolve: &F,
        matches: &mut Vec<Match>,
    ) where
        F: Fn(u32, &mut [*const u8; N]) -> Option<(usize, u16)>,
    {
        let mut scratch = ResolvedScratch::default();
        self.match_range_resolved_into_with(range, resolve, matches, &mut scratch);
    }

    /// Slice-based form of [`Matcher::match_range_resolved`]: resolves
    /// `items[i]` for each index.
    pub fn match_list_resolved<T, F, const N: usize>(
        &mut self,
        items: &[T],
        resolve: &F,
    ) -> Vec<Match>
    where
        F: Fn(&T, &mut [*const u8; N]) -> Option<(usize, u16)>,
    {
        self.match_range_resolved(items.len(), &|index, buf| {
            resolve(&items[index as usize], buf)
        })
    }

    /// Slice-based form of [`Matcher::match_range_resolved_into`]: item `i`
    /// is reported with index `i + item_index_offset`.
    pub fn match_list_resolved_into<T, F, const N: usize>(
        &mut self,
        items: &[T],
        item_index_offset: u32,
        resolve: &F,
        matches: &mut Vec<Match>,
    ) where
        F: Fn(&T, &mut [*const u8; N]) -> Option<(usize, u16)>,
    {
        Self::guard_against_haystack_overflow(items.len(), item_index_offset);
        let end = item_index_offset + items.len() as u32;
        self.match_range_resolved_into(
            item_index_offset..end,
            &|index, buf| resolve(&items[(index - item_index_offset) as usize], buf),
            matches,
        );
    }

    /// Lower bound on the haystack byte length for anything to match, so items
    /// below it are skipped before being gathered: for a single pattern, the
    /// needle's char count minus its typo budget (typos are ignored by literal
    /// modes, so this stays a valid bound there too).
    fn resolved_min_haystack_len(&self) -> usize {
        match (&self.patterns, self.raw_patterns.as_slice()) {
            (CompiledPatterns::Single(compiled), [pattern]) if !compiled.negated => compiled
                .max_typos
                .map(|max| pattern.needle.chars().count().saturating_sub(max as usize))
                .unwrap_or(0),
            _ => 0,
        }
    }

    pub(super) fn match_range_resolved_into_with<F, const N: usize>(
        &mut self,
        range: Range<u32>,
        resolve: &F,
        matches: &mut Vec<Match>,
        scratch: &mut ResolvedScratch,
    ) where
        F: Fn(u32, &mut [*const u8; N]) -> Option<(usize, u16)>,
    {
        let mut chunk_ptrs = [core::ptr::null::<u8>(); N];

        // Empty patterns match every resolvable item
        if self.patterns.is_empty() {
            matches.extend(
                range
                    .filter(|&index| resolve(index, &mut chunk_ptrs).is_some())
                    .map(|index| Match::from_index(index as usize)),
            );
            return;
        }

        let min_haystack_len = self.resolved_min_haystack_len();
        let ResolvedScratch { bytes, spans, hits } = scratch;

        let mut batch_start = range.start;
        while batch_start < range.end {
            let batch_end = batch_start
                .saturating_add(GATHER_BATCH as u32)
                .min(range.end);
            bytes.clear();
            spans.clear();

            for index in batch_start..batch_end {
                let Some((chunk_count, byte_len)) = resolve(index, &mut chunk_ptrs) else {
                    continue;
                };
                let len = byte_len as usize;
                if len < min_haystack_len {
                    continue;
                }
                debug_assert!(
                    chunk_count == len.div_ceil(SIMD_CHUNK_BYTES),
                    "chunk_count {chunk_count} does not cover byte_len {len}"
                );
                debug_assert!(
                    chunk_count <= N,
                    "chunk_count {chunk_count} exceeds capacity {N}"
                );

                // Gather whole chunks: fixed-size copies compile to plain vector
                // loads/stores instead of a memcpy call per chunk
                let start = bytes.len();
                bytes.reserve(chunk_count * SIMD_CHUNK_BYTES);
                // SAFETY: `reserve` guarantees `chunk_count * SIMD_CHUNK_BYTES` writable bytes
                // past `start`, and the caller guarantees each chunk pointer is readable for
                // `SIMD_CHUNK_BYTES` bytes (see the pointer contract). Only the first `len`
                // bytes are exposed via `set_len`.
                unsafe {
                    let dst = bytes
                        .as_mut_ptr()
                        .add(start)
                        .cast::<[u8; SIMD_CHUNK_BYTES]>();
                    for (chunk, &ptr) in chunk_ptrs[..chunk_count].iter().enumerate() {
                        dst.add(chunk)
                            .write_unaligned(ptr.cast::<[u8; SIMD_CHUNK_BYTES]>().read_unaligned());
                    }
                    bytes.set_len(start + len);
                }

                spans.push((index, start as u32, len as u32));
            }

            batch_start = batch_end;
            if spans.is_empty() {
                continue;
            }

            let haystacks: Vec<&str> = spans
                .iter()
                .map(|&(_, start, len)| {
                    let slice = &bytes[start as usize..(start + len) as usize];
                    // SAFETY: the caller guarantees the gathered bytes are valid UTF-8 (see the
                    // pointer contract)
                    unsafe { core::str::from_utf8_unchecked(slice) }
                })
                .collect();

            hits.clear();
            self.match_list_into(&haystacks, 0, hits);
            // Backends emit matches in input order, so `hit.index` is the position of the
            // gathered haystack; map it back to the item index
            matches.extend(hits.drain(..).map(|mut hit| {
                hit.index = spans[hit.index as usize].0;
                hit
            }));
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{Config, Matching, Pattern, SortStrategy};
    use alloc::{format, string::String, vec};

    /// Chunked haystack test item. Raw pointers are not `Sync`, so an arena-based caller
    /// wraps them in a type that guarantees the backing memory is immutable and alive.
    #[derive(Clone)]
    pub(crate) struct ChunkItem {
        ptrs: Vec<*const u8>,
        chunk_count: usize,
        byte_len: u16,
    }
    unsafe impl Sync for ChunkItem {}

    /// Splits a string into leaked 16-byte zero-padded chunks, as an arena-based caller would
    /// provide them to the resolved matching APIs.
    pub(crate) fn string_to_chunks(s: &str) -> ChunkItem {
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
        core::mem::forget(arena);
        ChunkItem {
            ptrs,
            chunk_count: n_chunks,
            byte_len: bytes.len() as u16,
        }
    }

    pub(crate) fn resolve_chunks<const N: usize>(
        item: &ChunkItem,
        ptrs_buf: &mut [*const u8; N],
    ) -> Option<(usize, u16)> {
        ptrs_buf[..item.ptrs.len()].copy_from_slice(&item.ptrs);
        Some((item.chunk_count, item.byte_len))
    }

    /// Resolved matching must produce the same matches as contiguous matching for arbitrary
    /// needle/haystack pairs, across typo budgets and sort strategies.
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
            proptest::bool::ANY,                                  // sort by score
        );

        runner
            .run(&strategy, |(needle, haystacks, max_typos, by_score)| {
                let sort = if by_score {
                    SortStrategy::ScoreThenIndexAsc
                } else {
                    SortStrategy::IndexAsc
                };
                let config = Config::default().max_typos(Some(max_typos)).sort(sort);

                let contiguous = Matcher::new(needle.as_str(), &config).match_list(&haystacks);

                let chunk_data: Vec<ChunkItem> =
                    haystacks.iter().map(|s| string_to_chunks(s)).collect();
                let resolved = Matcher::new(needle.as_str(), &config)
                    .match_list_resolved(&chunk_data, &resolve_chunks::<8>);

                prop_assert_eq!(
                    &contiguous,
                    &resolved,
                    "needle={:?} max_typos={} sort={:?}",
                    needle,
                    max_typos,
                    sort,
                );
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn range_api_matches_slice_api() {
        let haystacks: Vec<String> = (0..GATHER_BATCH + 5)
            .map(|i| {
                if i % 7 == 0 {
                    format!("abc_{i}")
                } else {
                    format!("zzz_{i}")
                }
            })
            .collect();
        let chunk_data: Vec<ChunkItem> = haystacks.iter().map(|s| string_to_chunks(s)).collect();
        let config = Config::default().sort(SortStrategy::IndexAsc);

        let by_slice =
            Matcher::new("abc", &config).match_list_resolved(&chunk_data, &resolve_chunks::<2>);
        let by_index = Matcher::new("abc", &config)
            .match_range_resolved(chunk_data.len(), &|index, buf: &mut [*const u8; 2]| {
                resolve_chunks(&chunk_data[index as usize], buf)
            });
        assert_eq!(by_slice, by_index);
        assert!(!by_index.is_empty());

        // Sub-range keeps global indices
        let mut partial = Vec::new();
        Matcher::new("abc", &config).match_range_resolved_into(
            7..(GATHER_BATCH as u32 + 1),
            &|index, buf: &mut [*const u8; 2]| resolve_chunks(&chunk_data[index as usize], buf),
            &mut partial,
        );
        let expected: Vec<_> = by_index
            .iter()
            .copied()
            .filter(|m| (7..GATHER_BATCH as u32 + 1).contains(&m.index))
            .collect();
        assert_eq!(partial, expected);
    }

    #[test]
    fn resolved_matches_contiguous_across_gather_batches() {
        // Larger than one gather batch, so indices must be remapped across batch boundaries
        let haystacks: Vec<String> = (0..3 * GATHER_BATCH + 7)
            .map(|i| {
                if i % 97 == 0 {
                    format!("src/abc_{i}.rs")
                } else {
                    format!("nomatch-{i}")
                }
            })
            .collect();
        let chunk_data: Vec<ChunkItem> = haystacks.iter().map(|s| string_to_chunks(s)).collect();

        for sort in [SortStrategy::ScoreThenIndexAsc, SortStrategy::IndexDesc] {
            let config = Config::default().sort(sort);
            let contiguous = Matcher::new("abc", &config).match_list(&haystacks);
            let resolved =
                Matcher::new("abc", &config).match_list_resolved(&chunk_data, &resolve_chunks::<8>);
            assert_eq!(contiguous, resolved, "sort={sort:?}");
            assert!(!resolved.is_empty());
        }
    }

    #[test]
    fn resolved_supports_literal_and_multi_pattern() {
        let haystacks = ["foo/bar", "bar/foo", "foo", "foobar", "qux"];
        let chunk_data: Vec<ChunkItem> = haystacks.iter().map(|s| string_to_chunks(s)).collect();
        let config = Config::default().sort(SortStrategy::IndexAsc);

        // Multi-pattern with negation
        let patterns = Pattern::parse_query("foo !^bar");
        let contiguous = Matcher::from_patterns(&patterns, &config).match_list(&haystacks);
        let resolved = Matcher::from_patterns(&patterns, &config)
            .match_list_resolved(&chunk_data, &resolve_chunks::<2>);
        assert_eq!(contiguous, resolved);
        assert_eq!(
            resolved.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 2, 3]
        );

        // Literal matching modes
        for matching in [
            Matching::Exact,
            Matching::Prefix,
            Matching::Suffix,
            Matching::Substring,
        ] {
            let config = config.matching(matching);
            let contiguous = Matcher::new("foo", &config).match_list(&haystacks);
            let resolved =
                Matcher::new("foo", &config).match_list_resolved(&chunk_data, &resolve_chunks::<2>);
            assert_eq!(contiguous, resolved, "matching={matching:?}");
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

        let config = Config::default().sort(SortStrategy::IndexAsc);
        let matches = Matcher::new("hw", &config).match_list_resolved(&items, &resolve);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 2]
        );

        // Empty needle reports every resolvable item
        let matches = Matcher::new("", &config).match_list_resolved(&items, &resolve);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 2]
        );

        // Index offsets are applied to skipped and matched items alike
        let mut matches = Vec::new();
        Matcher::new("hw", &config).match_list_resolved_into(&items, 10, &resolve, &mut matches);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![10, 12]
        );
    }
}
