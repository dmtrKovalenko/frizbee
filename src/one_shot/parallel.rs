use itertools::Itertools;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::Matcher;
use crate::sort::radix_sort_matches;
use crate::{Config, Match, Matchable, MatchableChunked};

macro_rules! worker_resolved_inner {
    ($matcher:expr, $items:expr, $resolve:expr, $next_chunk:expr, $num_chunks:expr, $chunk_size:expr, $sort:expr) => {{
        let mut local_matches = Vec::new();
        let needle = $matcher.needle.as_bytes();
        let min_haystack_len = $matcher
            .config
            .max_typos
            .map(|max| needle.len().saturating_sub(max as usize))
            .unwrap_or(0);
        let mut ptrs_buf = [core::ptr::null::<u8>(); 32];

        loop {
            let chunk_idx = $next_chunk.fetch_add(1, Ordering::Relaxed);
            if chunk_idx >= $num_chunks {
                break;
            }
            let start = chunk_idx * $chunk_size;
            let end = (start + $chunk_size).min($items.len());
            let items_chunk = &$items[start..end];

            for (index, item) in items_chunk.iter().enumerate() {
                let Some((chunk_count, byte_len)) = ($resolve)(item, &mut ptrs_buf) else {
                    continue;
                };

                let total_len = byte_len as usize;
                if total_len < min_haystack_len {
                    continue;
                }

                let chunk_ptrs = &ptrs_buf[..chunk_count];

                if let Some(match_) = $matcher.match_one_chunked(
                    chunk_ptrs,
                    byte_len,
                    (index as u32) + start as u32,
                ) {
                    local_matches.push(match_);
                }
            }
        }
        if $sort {
            radix_sort_matches(&mut local_matches);
        }
        local_matches
    }};
}

/// Per-thread worker body for resolved matching — AVX2 variant.
/// Destructures the Matcher's enums and calls AVX2 variants directly,
/// bypassing enum dispatch entirely so intrinsics inline as single instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn worker_resolved_avx2<T, F>(
    matcher: &mut Matcher,
    items: &[T],
    resolve: &F,
    next_chunk: &AtomicUsize,
    num_chunks: usize,
    chunk_size: usize,
    sort: bool,
) -> Vec<Match>
where
    F: Fn(&T, &mut [*const u8; 32]) -> Option<(usize, u16)>,
{
    use crate::prefilter::Prefilter;
    use crate::smith_waterman::simd::SmithWatermanMatcher;

    let max_typos = matcher.config.max_typos;
    let needle = matcher.needle.as_bytes();
    let min_haystack_len = max_typos
        .map(|max| needle.len().saturating_sub(max as usize))
        .unwrap_or(0);

    // Extract the concrete AVX2 variants — no enum dispatch in the hot loop
    let prefilter_avx = match &matcher.prefilter {
        Prefilter::AVX(p) => p,
        _ => {
            // Fallback if somehow not AVX2
            return worker_resolved_inner!(matcher, items, resolve, next_chunk, num_chunks, chunk_size, sort);
        }
    };
    let sw_avx = match &mut matcher.smith_waterman {
        SmithWatermanMatcher::AVX2(sw) => sw,
        _ => {
            return worker_resolved_inner!(matcher, items, resolve, next_chunk, num_chunks, chunk_size, sort);
        }
    };

    let mut local_matches = Vec::new();
    let mut ptrs_buf = [core::ptr::null::<u8>(); 32];

    loop {
        let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
        if chunk_idx >= num_chunks {
            break;
        }
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(items.len());
        let items_chunk = &items[start..end];

        for (index, item) in items_chunk.iter().enumerate() {
            let Some((chunk_count, byte_len)) = resolve(item, &mut ptrs_buf) else {
                continue;
            };

            if (byte_len as usize) < min_haystack_len {
                continue;
            }

            let chunk_ptrs = &ptrs_buf[..chunk_count];

            // Direct AVX2 prefilter — no enum dispatch
            let (prefilter_passed, skipped_chunks) = max_typos.map_or((true, 0), |mt| {
                prefilter_avx.match_haystack_typos_chunked(chunk_ptrs, byte_len, mt)
            });
            if !prefilter_passed {
                continue;
            }

            let chunk_ptrs = &chunk_ptrs[skipped_chunks..];
            let byte_len = byte_len - (skipped_chunks as u16 * 16);

            // Direct AVX2 smith-waterman — no enum dispatch
            #[cfg(feature = "match_end_col")]
            let result = sw_avx.match_haystack_chunked_with_end_col(chunk_ptrs, byte_len, max_typos);
            #[cfg(not(feature = "match_end_col"))]
            let result = sw_avx.match_haystack_chunked(chunk_ptrs, byte_len, max_typos);

            #[cfg(feature = "match_end_col")]
            if let Some((mut score, end_col)) = result {
                let exact = chunk_ptrs.len() == 1 && (byte_len as usize) == needle.len() && {
                    let haystack = core::slice::from_raw_parts(chunk_ptrs[0], byte_len as usize);
                    needle == haystack
                };
                if exact {
                    score += matcher.config.scoring.exact_match_bonus;
                }
                local_matches.push(Match {
                    index: (index as u32) + start as u32,
                    score,
                    exact,
                    end_col,
                });
            }

            #[cfg(not(feature = "match_end_col"))]
            if let Some(mut score) = result {
                let exact = chunk_ptrs.len() == 1 && (byte_len as usize) == needle.len() && {
                    let haystack = core::slice::from_raw_parts(chunk_ptrs[0], byte_len as usize);
                    needle == haystack
                };
                if exact {
                    score += matcher.config.scoring.exact_match_bonus;
                }
                local_matches.push(Match {
                    index: (index as u32) + start as u32,
                    score,
                    exact,
                });
            }
        }
    }

    if sort {
        radix_sort_matches(&mut local_matches);
    }
    local_matches
}

/// Per-thread worker body for resolved matching — SSE4.1 variant.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3", enable = "sse4.1")]
unsafe fn worker_resolved_sse<T, F>(
    matcher: &mut Matcher,
    items: &[T],
    resolve: &F,
    next_chunk: &AtomicUsize,
    num_chunks: usize,
    chunk_size: usize,
    sort: bool,
) -> Vec<Match>
where
    F: Fn(&T, &mut [*const u8; 32]) -> Option<(usize, u16)>,
{
    worker_resolved_inner!(matcher, items, resolve, next_chunk, num_chunks, chunk_size, sort)
}


pub fn match_list_parallel<S1: AsRef<str>, S2: Matchable + Sync>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
    threads: usize,
) -> Vec<Match> {
    assert!(
        haystacks.len() < (u32::MAX as usize),
        "haystack index overflow"
    );

    if needle.as_ref().is_empty() {
        return haystacks
            .iter()
            .enumerate()
            .filter(|(_, item)| item.match_str().is_some())
            .map(|(index, _)| Match {
                index: index as u32,
                score: 0,
                exact: false,
                #[cfg(feature = "match_end_col")]
                end_col: 0,
            })
            .collect();
    }

    if haystacks.is_empty() {
        return vec![];
    }

    // Smaller chunks enable better load balancing via stealing
    // but too small increases atomic contention
    let chunk_size = 512;
    let num_chunks = haystacks.len().div_ceil(chunk_size);
    let next_chunk = AtomicUsize::new(0);

    let needle = needle.as_ref();
    let matcher = Matcher::new(needle, config);

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

                        matcher.match_list_into(haystacks_chunk, start as u32, &mut local_matches);
                    }

                    // Each thread sorts so that we can perform k-way merge
                    if config.sort {
                        radix_sort_matches(&mut local_matches);
                    }

                    local_matches
                })
            })
            .collect();

        if config.sort {
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .kmerge()
                .collect()
        } else {
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        }
    })
}

/// Parallel chunked matching with a resolver callback.
///
/// For each item, `resolve` fills a stack buffer with chunk pointers and returns
/// `Some((chunk_count, byte_len))` or `None` to skip the item.
pub fn match_list_parallel_resolved<
    S1: AsRef<str>,
    T: Sync,
    F: Fn(&T, &mut [*const u8; 32]) -> Option<(usize, u16)> + Sync,
>(
    needle: S1,
    items: &[T],
    resolve: &F,
    config: &Config,
    threads: usize,
) -> Vec<Match> {
    assert!(items.len() < (u32::MAX as usize), "item index overflow");

    if needle.as_ref().is_empty() {
        let mut ptrs_buf = [core::ptr::null::<u8>(); 32];
        return items
            .iter()
            .enumerate()
            .filter(|(_, item)| resolve(item, &mut ptrs_buf).is_some())
            .map(|(index, _)| Match {
                index: index as u32,
                score: 0,
                exact: false,
                #[cfg(feature = "match_end_col")]
                end_col: 0,
            })
            .collect();
    }

    if items.is_empty() {
        return vec![];
    }

    let chunk_size = 512;
    let num_chunks = items.len().div_ceil(chunk_size);
    let next_chunk = AtomicUsize::new(0);

    let needle = needle.as_ref();
    let matcher = Matcher::new(needle, config);

    thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut matcher = matcher.clone();

                    #[cfg(target_arch = "x86_64")]
                    {
                        if std::arch::is_x86_feature_detected!("avx2") {
                            return unsafe {
                                worker_resolved_avx2(
                                    &mut matcher,
                                    items,
                                    resolve,
                                    &next_chunk,
                                    num_chunks,
                                    chunk_size,
                                    config.sort,
                                )
                            };
                        }
                        if std::arch::is_x86_feature_detected!("sse4.1") {
                            return unsafe {
                                worker_resolved_sse(
                                    &mut matcher,
                                    items,
                                    resolve,
                                    &next_chunk,
                                    num_chunks,
                                    chunk_size,
                                    config.sort,
                                )
                            };
                        }
                    }

                    worker_resolved_inner!(
                        matcher,
                        items,
                        resolve,
                        next_chunk,
                        num_chunks,
                        chunk_size,
                        config.sort
                    )
                })
            })
            .collect();

        if config.sort {
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .kmerge()
                .collect()
        } else {
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        }
    })
}

pub fn match_list_parallel_chunked<S1: AsRef<str>, C: MatchableChunked + Sync>(
    needle: S1,
    haystacks: &[C],
    ctx: &C::Ctx,
    config: &Config,
    threads: usize,
) -> Vec<Match>
where
    C::Ctx: Sync,
{
    assert!(
        haystacks.len() < (u32::MAX as usize),
        "haystack index overflow"
    );

    if needle.as_ref().is_empty() {
        return haystacks
            .iter()
            .enumerate()
            .filter(|(_, item)| item.haystack_info(ctx).is_some())
            .map(|(index, _)| Match {
                index: index as u32,
                score: 0,
                exact: false,
                #[cfg(feature = "match_end_col")]
                end_col: 0,
            })
            .collect();
    }

    if haystacks.is_empty() {
        return vec![];
    }

    let chunk_size = 512;
    let num_chunks = haystacks.len().div_ceil(chunk_size);
    let next_chunk = AtomicUsize::new(0);

    let needle = needle.as_ref();
    let matcher = Matcher::new(needle, config);

    thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut local_matches = Vec::new();
                    let mut matcher = matcher.clone();

                    loop {
                        let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
                        if chunk_idx >= num_chunks {
                            break;
                        }

                        let start = chunk_idx * chunk_size;
                        let end = (start + chunk_size).min(haystacks.len());
                        let haystacks_chunk = &haystacks[start..end];

                        matcher.match_list_chunked_into(
                            haystacks_chunk,
                            ctx,
                            start as u32,
                            &mut local_matches,
                        );
                    }

                    if config.sort {
                        radix_sort_matches(&mut local_matches);
                    }

                    local_matches
                })
            })
            .collect();

        if config.sort {
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .kmerge()
                .collect()
        } else {
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        }
    })
}
