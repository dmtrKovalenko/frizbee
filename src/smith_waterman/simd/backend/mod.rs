//! SIMD backend abstraction for the Smith-Waterman kernel.
//!
//! Each backend exposes its native vector width via [`Backend::LANES`] and
//! three associated vector types: [`BytesVec`] for haystack bytes (LANES bytes
//! wide), [`MaskVec`] for boolean comparison results, and [`ScoreVec`] for the
//! score matrix (LANES × u16 or LANES × u8). Keeping masks as a distinct type
//! lets AVX-512 carry them as native `__mmask*` bitmasks rather than
//! 64-byte vectors, while other backends can alias [`MaskVec`] to their
//! [`BytesVec`] type with no loss.
//!
//! The kernel itself is generic over [`Backend`] and the lane count threads
//! through chunk arithmetic, the matrix stride, the alignment-path iterator,
//! and the horizontal-gap propagation unroll.
//!
//! Availabled backends:
//!   - AVX-512: LANES = 32/64 (scoring u16 x 32 = 512-bit or u8 x 64 = 512-bit)
//!   - AVX2:    LANES = 16/32 (scoring u16 x 16 = 256-bit or u8 x 32 = 256-bit)
//!   - SSE:     LANES = 8/16  (scoring u16 x 8 = 128-bit or u8 x 16 = 128-bit)
//!   - NEON:    LANES = 8/16  (scoring u16 x 8 = 128-bit or u8 x 16 = 128-bit)
//!   - Scalar:  LANES = 16/32 (fallback for non-SIMD systems)

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod sse;

#[cfg(target_arch = "x86_64")]
pub use avx::{BackendAVX, BackendAVXU8};
#[cfg(target_arch = "x86_64")]
pub use avx512::{BackendAVX512, BackendAVX512U8};
#[cfg(target_arch = "aarch64")]
pub use neon::{BackendNEON, BackendNEONU8};
pub use scalar::{BackendScalar8, BackendScalar16U8};
#[cfg(target_arch = "x86_64")]
pub use sse::{BackendSSE, BackendSSEU8};

/// A SIMD backend for Smith-Waterman matching that supports variable-width
/// lanes (8, 16, 32, 64) and score primitives (u8, u16).
///
/// `LANES` is the number of cells per chunk in the score matrix. The matching
/// [`BytesVec`] holds `LANES` meaningful bytes (one per lane); comparisons
/// produce a [`MaskVec`] which can be widened element-wise to the [`ScoreVec`]
/// via [`Backend::widen_mask`].
pub trait Backend: Sized + core::fmt::Debug + Clone + 'static {
    const LANES: usize;
    /// Size of a single score lane in bytes. 1 for u8-element backends,
    /// 2 for u16-element backends. Used by the alignment-path iterator to
    /// read individual cells from the matrix's raw byte view.
    const LANE_BYTES: usize;
    type Bytes: BytesVec<Mask = Self::Mask>;
    type Mask: MaskVec;
    type Score: ScoreVec;

    /// Whether this backend's required CPU features are available at runtime
    fn is_available() -> bool;

    /// Widen a comparison [`MaskVec`] into a [`ScoreVec`] with each lane set
    /// to `0xFF` (u8 score) or `0xFFFF` (u16 score) where the mask is true and
    /// zero elsewhere.
    ///
    /// # Safety
    /// The backend's required target features must be enabled at the call site.
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score;

    /// Propagate horizontal (left-direction) gaps across a score row.
    ///
    /// The number of unrolled stages is fixed per backend: 8-lane backends do
    /// 3 stages (shifts of 1, 2, 4 lanes), 16-lane backends do 4 stages (1, 2,
    /// 4, 8), 32-lane backends do 5 stages, and 64-lane backends do 6 stages.
    ///
    /// # Safety
    /// The backend's required target features must be enabled at the call site.
    unsafe fn propagate_horizontal_gaps(
        row: Self::Score,
        adjacent_row: Self::Score,
        match_mask: Self::Score,
        adjacent_match_mask: Self::Score,
        gap_open_penalty: Self::Score,
        gap_extend_penalty: Self::Score,
    ) -> Self::Score;
}

/// A byte-wide vector with `LANES` meaningful bytes
pub trait BytesVec: Copy + core::fmt::Debug {
    /// The backend's mask type, produced by comparison operations.
    type Mask: MaskVec;

    /// Broadcast `value` into every byte lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn splat(value: u8) -> Self;

    /// Per-lane equality, producing a true bit in each lane where equal.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn eq(self, other: Self) -> Self::Mask;

    /// Per-lane unsigned greater-than, producing a true bit where
    /// `self > other`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn gt(self, other: Self) -> Self::Mask;

    /// Per-lane unsigned less-than, producing a true bit where `self < other`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn lt(self, other: Self) -> Self::Mask;

    /// Load up to `LANES` bytes from `data + start`. `len` is the total length
    /// of the buffer pointed to by `data`.
    ///
    /// # Safety
    /// Caller must guarantee `data` points to at least `len` bytes and the
    /// backend's target features are enabled.
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self;

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8>;
}

/// A boolean mask over `LANES` lanes, produced by [`BytesVec`] comparisons.
///
/// Each lane holds a logical true/false. Non-AVX-512 backends typically alias
/// this to their [`BytesVec`] type and represent the mask as 0xFF/0x00 per
/// byte. AVX-512 uses a native `__mmask64` bitmask.
pub trait MaskVec: Copy + core::fmt::Debug {
    /// False in every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn zero() -> Self;

    /// Per-lane logical AND.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn and(self, other: Self) -> Self;

    /// Per-lane logical OR.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn or(self, other: Self) -> Self;

    /// Per-lane logical NOT.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn not(self) -> Self;

    /// Shift right by 1 lane, filling lane 0 with the highest meaningful lane
    /// of `prev`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self;

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool>;
}

/// A score-wide vector holding `LANES` u16 elements.
pub trait ScoreVec: Copy + core::fmt::Debug {
    /// Zero in every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn zero() -> Self;

    /// Broadcast `value` into every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn splat(value: u16) -> Self;

    /// `value` in lane 0, zero in all other lanes
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn first_lane(value: u16) -> Self;

    /// Per-lane unsigned max
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn max(self, other: Self) -> Self;

    /// Horizontal unsigned max across all lanes
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn horizontal_max(self) -> u16;

    /// Per-lane wrapping add
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn add(self, other: Self) -> Self;

    /// Per-lane saturating subtract (saturating at zero)
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn subs(self, other: Self) -> Self;

    /// Bitwise AND
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn and(self, other: Self) -> Self;

    /// Shift right by `L` lanes, filling the low `L` lanes with the high `L`
    /// lanes of `prev`. `L` must be in `0..=LANES`. `L == LANES` returns
    /// `prev` unmodified.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self;

    /// Index of the lowest lane equal to `search`, or `LANES` if absent
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn find_lane(self, search: u16) -> usize;

    // ----- Test-only helpers ---------------------------------------------

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16>;
}

// ---------------------------------------------------------------------------
// Gap propagation helpers shared by all backends.
//
// Each backend's Backend::propagate_horizontal_gaps calls one of these
// generic-over-B functions, selected by lane count. The macro hides the
// fully-qualified `<<B as Backend>::Score as ScoreVec>::*` boilerplate.
// ---------------------------------------------------------------------------

macro_rules! gap_step {
    ($B:ty, $shift:literal, $row:ident, $adj:ident, $mm:ident, $amm:ident, $gop:ident, $gex:ident) => {
        let shifted_row =
            <<$B as Backend>::Score as ScoreVec>::shift_right_padded::<$shift>($row, $adj);
        let shifted_match_mask =
            <<$B as Backend>::Score as ScoreVec>::shift_right_padded::<$shift>($mm, $amm);
        let gap_penalty = <<$B as Backend>::Score as ScoreVec>::add(
            $gex,
            <<$B as Backend>::Score as ScoreVec>::and($gop, shifted_match_mask),
        );
        let decayed = <<$B as Backend>::Score as ScoreVec>::subs(shifted_row, gap_penalty);
        let $row = <<$B as Backend>::Score as ScoreVec>::max($row, decayed);
        let $gex = <<$B as Backend>::Score as ScoreVec>::add($gex, $gex);
    };
}

#[inline(always)]
pub(crate) unsafe fn propagate_8_lane<B: Backend>(
    row: B::Score,
    adj: B::Score,
    mm: B::Score,
    amm: B::Score,
    gop: B::Score,
    gex: B::Score,
) -> B::Score {
    unsafe {
        gap_step!(B, 1, row, adj, mm, amm, gop, gex);
        gap_step!(B, 2, row, adj, mm, amm, gop, gex);
        gap_step!(B, 4, row, adj, mm, amm, gop, gex);
        let _ = gex;
        row
    }
}

#[inline(always)]
pub(crate) unsafe fn propagate_16_lane<B: Backend>(
    row: B::Score,
    adj: B::Score,
    mm: B::Score,
    amm: B::Score,
    gop: B::Score,
    gex: B::Score,
) -> B::Score {
    unsafe {
        gap_step!(B, 1, row, adj, mm, amm, gop, gex);
        gap_step!(B, 2, row, adj, mm, amm, gop, gex);
        gap_step!(B, 4, row, adj, mm, amm, gop, gex);
        gap_step!(B, 8, row, adj, mm, amm, gop, gex);
        let _ = gex;
        row
    }
}

#[inline(always)]
pub(crate) unsafe fn propagate_32_lane<B: Backend>(
    row: B::Score,
    adj: B::Score,
    mm: B::Score,
    amm: B::Score,
    gop: B::Score,
    gex: B::Score,
) -> B::Score {
    unsafe {
        gap_step!(B, 1, row, adj, mm, amm, gop, gex);
        gap_step!(B, 2, row, adj, mm, amm, gop, gex);
        gap_step!(B, 4, row, adj, mm, amm, gop, gex);
        gap_step!(B, 8, row, adj, mm, amm, gop, gex);
        gap_step!(B, 16, row, adj, mm, amm, gop, gex);
        let _ = gex;
        row
    }
}

#[inline(always)]
pub(crate) unsafe fn propagate_64_lane<B: Backend>(
    row: B::Score,
    adj: B::Score,
    mm: B::Score,
    amm: B::Score,
    gop: B::Score,
    gex: B::Score,
) -> B::Score {
    unsafe {
        gap_step!(B, 1, row, adj, mm, amm, gop, gex);
        gap_step!(B, 2, row, adj, mm, amm, gop, gex);
        gap_step!(B, 4, row, adj, mm, amm, gop, gex);
        gap_step!(B, 8, row, adj, mm, amm, gop, gex);
        gap_step!(B, 16, row, adj, mm, amm, gop, gex);
        gap_step!(B, 32, row, adj, mm, amm, gop, gex);
        let _ = gex;
        row
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scoring;
    use crate::smith_waterman::simd::{Kernel, SmithWaterman, score_fits_in_u8};
    use proptest::prelude::*;

    #[derive(Debug, Clone, Copy)]
    struct TestScalarBytes<const LANES: usize>([u8; LANES]);

    #[derive(Debug, Clone, Copy)]
    struct TestScalarScoreU16<const LANES: usize>([u16; LANES]);

    #[derive(Debug, Clone, Copy)]
    struct TestScalarScoreU8<const LANES: usize>([u8; LANES]);

    #[derive(Debug, Clone)]
    struct TestScalar16;

    #[derive(Debug, Clone)]
    struct TestScalar32;

    #[derive(Debug, Clone)]
    struct TestScalar32U8;

    #[derive(Debug, Clone)]
    struct TestScalar64U8;

    impl<const LANES: usize> BytesVec for TestScalarBytes<LANES> {
        type Mask = TestScalarBytes<LANES>;

        #[inline(always)]
        unsafe fn splat(value: u8) -> Self {
            Self([value; LANES])
        }

        #[inline(always)]
        unsafe fn eq(self, other: Self) -> Self::Mask {
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = if self.0[idx] == other.0[idx] {
                    u8::MAX
                } else {
                    0
                };
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn gt(self, other: Self) -> Self::Mask {
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = if self.0[idx] > other.0[idx] {
                    u8::MAX
                } else {
                    0
                };
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn lt(self, other: Self) -> Self::Mask {
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = if self.0[idx] < other.0[idx] {
                    u8::MAX
                } else {
                    0
                };
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
            let mut out = [0u8; LANES];
            let take = len.saturating_sub(start).min(LANES);
            for (idx, lane) in out.iter_mut().take(take).enumerate() {
                *lane = unsafe { *data.add(start + idx) };
            }
            Self(out)
        }

        #[cfg(test)]
        fn from_lanes(values: &[u8]) -> Self {
            assert_eq!(values.len(), LANES);
            let mut out = [0u8; LANES];
            out.copy_from_slice(values);
            Self(out)
        }

        #[cfg(test)]
        fn to_lanes(self) -> Vec<u8> {
            self.0.to_vec()
        }
    }

    impl<const LANES: usize> MaskVec for TestScalarBytes<LANES> {
        #[inline(always)]
        unsafe fn zero() -> Self {
            Self([0; LANES])
        }

        #[inline(always)]
        unsafe fn and(self, other: Self) -> Self {
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = self.0[idx] & other.0[idx];
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn or(self, other: Self) -> Self {
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = self.0[idx] | other.0[idx];
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn not(self) -> Self {
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = !self.0[idx];
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
            let mut out = [0u8; LANES];
            out[0] = prev.0[LANES - 1];
            out[1..LANES].copy_from_slice(&self.0[..(LANES - 1)]);
            Self(out)
        }

        #[cfg(test)]
        fn from_lanes(values: &[bool]) -> Self {
            assert_eq!(values.len(), LANES);
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = if values[idx] { u8::MAX } else { 0 };
            }
            Self(out)
        }

        #[cfg(test)]
        fn to_lanes(self) -> Vec<bool> {
            self.0.iter().map(|&lane| lane != 0).collect()
        }
    }

    impl<const LANES: usize> ScoreVec for TestScalarScoreU16<LANES> {
        #[inline(always)]
        unsafe fn zero() -> Self {
            Self([0; LANES])
        }

        #[inline(always)]
        unsafe fn splat(value: u16) -> Self {
            Self([value; LANES])
        }

        #[inline(always)]
        unsafe fn first_lane(value: u16) -> Self {
            let mut out = [0u16; LANES];
            out[0] = value;
            Self(out)
        }

        #[inline(always)]
        unsafe fn max(self, other: Self) -> Self {
            let mut out = [0u16; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = self.0[idx].max(other.0[idx]);
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn horizontal_max(self) -> u16 {
            *self.0.iter().max().unwrap()
        }

        #[inline(always)]
        unsafe fn add(self, other: Self) -> Self {
            let mut out = [0u16; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = self.0[idx].wrapping_add(other.0[idx]);
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn subs(self, other: Self) -> Self {
            let mut out = [0u16; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = self.0[idx].saturating_sub(other.0[idx]);
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn and(self, other: Self) -> Self {
            let mut out = [0u16; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = self.0[idx] & other.0[idx];
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
            let shift = L as usize;
            debug_assert!(shift <= LANES);
            let mut out = [0u16; LANES];
            for idx in 0..shift {
                out[idx] = prev.0[LANES - shift + idx];
            }
            out[shift..LANES].copy_from_slice(&self.0[..(LANES - shift)]);
            Self(out)
        }

        #[inline(always)]
        unsafe fn find_lane(self, search: u16) -> usize {
            for (idx, &lane) in self.0.iter().enumerate() {
                if lane == search {
                    return idx;
                }
            }
            LANES
        }

        #[cfg(test)]
        fn from_lanes(values: &[u16]) -> Self {
            assert_eq!(values.len(), LANES);
            let mut out = [0u16; LANES];
            out.copy_from_slice(values);
            Self(out)
        }

        #[cfg(test)]
        fn to_lanes(self) -> Vec<u16> {
            self.0.to_vec()
        }
    }

    impl<const LANES: usize> ScoreVec for TestScalarScoreU8<LANES> {
        #[inline(always)]
        unsafe fn zero() -> Self {
            Self([0; LANES])
        }

        #[inline(always)]
        unsafe fn splat(value: u16) -> Self {
            Self([value as u8; LANES])
        }

        #[inline(always)]
        unsafe fn first_lane(value: u16) -> Self {
            let mut out = [0u8; LANES];
            out[0] = value as u8;
            Self(out)
        }

        #[inline(always)]
        unsafe fn max(self, other: Self) -> Self {
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = self.0[idx].max(other.0[idx]);
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn horizontal_max(self) -> u16 {
            *self.0.iter().max().unwrap() as u16
        }

        #[inline(always)]
        unsafe fn add(self, other: Self) -> Self {
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = self.0[idx].wrapping_add(other.0[idx]);
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn subs(self, other: Self) -> Self {
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = self.0[idx].saturating_sub(other.0[idx]);
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn and(self, other: Self) -> Self {
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = self.0[idx] & other.0[idx];
            }
            Self(out)
        }

        #[inline(always)]
        unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
            let shift = L as usize;
            debug_assert!(shift <= LANES);
            let mut out = [0u8; LANES];
            for idx in 0..shift {
                out[idx] = prev.0[LANES - shift + idx];
            }
            out[shift..LANES].copy_from_slice(&self.0[..(LANES - shift)]);
            Self(out)
        }

        #[inline(always)]
        unsafe fn find_lane(self, search: u16) -> usize {
            let search = search as u8;
            for (idx, &lane) in self.0.iter().enumerate() {
                if lane == search {
                    return idx;
                }
            }
            LANES
        }

        #[cfg(test)]
        fn from_lanes(values: &[u16]) -> Self {
            assert_eq!(values.len(), LANES);
            let mut out = [0u8; LANES];
            for (idx, lane) in out.iter_mut().enumerate() {
                *lane = values[idx] as u8;
            }
            Self(out)
        }

        #[cfg(test)]
        fn to_lanes(self) -> Vec<u16> {
            self.0.iter().map(|&lane| lane as u16).collect()
        }
    }

    macro_rules! test_scalar_backend {
        ($backend:ty, $lanes:literal, $lane_bytes:literal, $score:ty, $propagate:ident) => {
            impl Backend for $backend {
                const LANES: usize = $lanes;
                const LANE_BYTES: usize = $lane_bytes;
                type Bytes = TestScalarBytes<$lanes>;
                type Mask = TestScalarBytes<$lanes>;
                type Score = $score;

                fn is_available() -> bool {
                    true
                }

                #[inline(always)]
                unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
                    let mut out = [0u16; $lanes];
                    let full_mask = if $lane_bytes == 1 {
                        u8::MAX as u16
                    } else {
                        u16::MAX
                    };
                    for (idx, lane) in out.iter_mut().enumerate() {
                        *lane = if m.0[idx] != 0 { full_mask } else { 0 };
                    }
                    Self::Score::from_lanes(&out)
                }

                #[inline(always)]
                unsafe fn propagate_horizontal_gaps(
                    row: Self::Score,
                    adjacent_row: Self::Score,
                    match_mask: Self::Score,
                    adjacent_match_mask: Self::Score,
                    gap_open_penalty: Self::Score,
                    gap_extend_penalty: Self::Score,
                ) -> Self::Score {
                    unsafe {
                        super::$propagate::<Self>(
                            row,
                            adjacent_row,
                            match_mask,
                            adjacent_match_mask,
                            gap_open_penalty,
                            gap_extend_penalty,
                        )
                    }
                }
            }
        };
    }

    test_scalar_backend!(
        TestScalar16,
        16,
        2,
        TestScalarScoreU16<16>,
        propagate_16_lane
    );
    test_scalar_backend!(
        TestScalar32,
        32,
        2,
        TestScalarScoreU16<32>,
        propagate_32_lane
    );
    test_scalar_backend!(
        TestScalar32U8,
        32,
        1,
        TestScalarScoreU8<32>,
        propagate_32_lane
    );
    test_scalar_backend!(
        TestScalar64U8,
        64,
        1,
        TestScalarScoreU8<64>,
        propagate_64_lane
    );

    // ----- BytesVec property tests ---------------------------------------

    fn check_bytes_splat<B: Backend>() {
        unsafe {
            let v = B::Bytes::splat(0x42);
            assert_eq!(BytesVec::to_lanes(v), vec![0x42; B::LANES]);
        }
    }

    fn check_bytes_eq<B: Backend>() {
        unsafe {
            let a_in = (0u8..(B::LANES as u8)).collect::<Vec<_>>();
            let b_in = a_in
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { v.wrapping_add(1) })
                .collect::<Vec<_>>();
            let a = B::Bytes::from_lanes(&a_in);
            let b = B::Bytes::from_lanes(&b_in);
            let result = MaskVec::to_lanes(a.eq(b));
            for (i, r) in result.iter().enumerate() {
                assert_eq!(*r, i % 2 == 0, "lane {i}");
            }
        }
    }

    fn check_bytes_gt_lt<B: Backend>() {
        unsafe {
            let a_in = vec![5u8; B::LANES];
            let mut b_in = vec![5u8; B::LANES];
            if B::LANES >= 2 {
                b_in[0] = 4;
                b_in[1] = 6;
            }
            let a = B::Bytes::from_lanes(&a_in);
            let b = B::Bytes::from_lanes(&b_in);
            let gt = MaskVec::to_lanes(a.gt(b));
            let lt = MaskVec::to_lanes(a.lt(b));
            if B::LANES >= 2 {
                assert!(gt[0]);
                assert!(!gt[1]);
                assert!(!lt[0]);
                assert!(lt[1]);
            }
        }
    }

    fn check_mask_and_or_not<B: Backend>() {
        unsafe {
            let pattern_a: Vec<bool> = (0..B::LANES).map(|i| i % 2 == 0).collect();
            let pattern_b: Vec<bool> = (0..B::LANES).map(|i| i % 3 == 0).collect();
            let a = B::Mask::from_lanes(&pattern_a);
            let b = B::Mask::from_lanes(&pattern_b);
            let and = a.and(b).to_lanes();
            let or = a.or(b).to_lanes();
            let not_a = a.not().to_lanes();
            for i in 0..B::LANES {
                assert_eq!(and[i], pattern_a[i] && pattern_b[i], "and lane {i}");
                assert_eq!(or[i], pattern_a[i] || pattern_b[i], "or lane {i}");
                assert_eq!(not_a[i], !pattern_a[i], "not lane {i}");
            }
        }
    }

    fn check_mask_zero<B: Backend>() {
        unsafe {
            let z = B::Mask::zero();
            assert_eq!(MaskVec::to_lanes(z), vec![false; B::LANES]);
        }
    }

    fn check_bytes_load_partial<B: Backend>() {
        unsafe {
            let data: Vec<u8> = (1..=64).collect();
            let lanes = B::LANES;
            for start in (0..32).step_by(lanes) {
                for len in (start + 1)..=32 {
                    let v = B::Bytes::load_partial(data.as_ptr(), start, len);
                    let got = BytesVec::to_lanes(v);
                    let mut expected = vec![0u8; lanes];
                    let take = (len - start).min(lanes);
                    expected[..take].copy_from_slice(&data[start..start + take]);
                    assert_eq!(got, expected, "start={start} len={len}");
                }
            }
        }
    }

    fn check_mask_shift_right_padded_1<B: Backend>() {
        unsafe {
            let a_in: Vec<bool> = (0..B::LANES).map(|i| i % 2 == 0).collect();
            let p_in: Vec<bool> = (0..B::LANES).map(|i| i % 3 == 0).collect();
            let a = B::Mask::from_lanes(&a_in);
            let p = B::Mask::from_lanes(&p_in);
            let got = MaskVec::to_lanes(a.shift_right_padded_1(p));

            let mut expected = vec![false; B::LANES];
            expected[0] = p_in[B::LANES - 1];
            expected[1..B::LANES].copy_from_slice(&a_in[0..(B::LANES - 1)]);
            assert_eq!(got, expected);
        }
    }

    // ----- ScoreVec property tests ---------------------------------------

    fn check_score_zero<B: Backend>() {
        unsafe {
            assert_eq!(B::Score::zero().to_lanes(), vec![0u16; B::LANES]);
        }
    }

    fn check_score_splat<B: Backend>() {
        // Use a value that fits in u8 so the same test applies to both
        // u8- and u16-element backends.
        unsafe {
            assert_eq!(B::Score::splat(0xAB).to_lanes(), vec![0xABu16; B::LANES]);
        }
    }

    fn check_score_first_lane<B: Backend>() {
        unsafe {
            let v = B::Score::first_lane(0xCD);
            let lanes = v.to_lanes();
            assert_eq!(lanes[0], 0xCD);
            for &l in &lanes[1..] {
                assert_eq!(l, 0);
            }
        }
    }

    fn check_score_max_add_subs<B: Backend>() {
        // Values stay in u8 range so the same inputs work for both widths.
        unsafe {
            let a_in: Vec<u16> = (0..B::LANES).map(|i| (i % 100) as u16).collect();
            let b_in: Vec<u16> = (0..B::LANES).map(|i| (B::LANES - i) as u16 % 50).collect();
            let a = B::Score::from_lanes(&a_in);
            let b = B::Score::from_lanes(&b_in);
            let max = a.max(b).to_lanes();
            for i in 0..B::LANES {
                assert_eq!(max[i], a_in[i].max(b_in[i]));
            }
            let added = a.add(b).to_lanes();
            for i in 0..B::LANES {
                assert_eq!(added[i], a_in[i].wrapping_add(b_in[i]));
            }
            let subbed = a.subs(b).to_lanes();
            for i in 0..B::LANES {
                assert_eq!(subbed[i], a_in[i].saturating_sub(b_in[i]));
            }
        }
    }

    fn check_score_horizontal_max<B: Backend>() {
        unsafe {
            let mut a_in = vec![10u16; B::LANES];
            a_in[B::LANES / 2] = 222;
            let a = B::Score::from_lanes(&a_in);
            assert_eq!(a.horizontal_max(), 222);

            let zero = B::Score::zero();
            assert_eq!(zero.horizontal_max(), 0);
        }
    }

    fn check_score_find_lane<B: Backend>() {
        unsafe {
            // For 32 lanes, step of 7 stays under 224 — fits in u8.
            let a_in: Vec<u16> = (0..B::LANES).map(|i| (i as u16) * 7).collect();
            let a = B::Score::from_lanes(&a_in);
            for (i, &v) in a_in.iter().enumerate() {
                assert_eq!(a.find_lane(v), i);
            }
            // Search for a value that can't appear in the data; works for
            // both u8 and u16 backends.
            assert_eq!(a.find_lane(251), B::LANES);
        }
    }

    fn check_score_shift_right_padded_each<B: Backend, const L: i32>() {
        unsafe {
            // Distinct values for `a` and `p` that all fit in u8.
            let a_in: Vec<u16> = (0..B::LANES).map(|i| (i as u16) + 1).collect();
            let p_in: Vec<u16> = (0..B::LANES).map(|i| (i as u16) + 100).collect();
            let a = B::Score::from_lanes(&a_in);
            let p = B::Score::from_lanes(&p_in);
            let got = a.shift_right_padded::<L>(p).to_lanes();

            let n = L as usize;
            let mut expected = vec![0u16; B::LANES];
            for i in 0..n {
                expected[i] = p_in[B::LANES - n + i];
            }
            expected[n..B::LANES].copy_from_slice(&a_in[0..(B::LANES - n)]);
            assert_eq!(got, expected, "L = {}", L);
        }
    }

    fn check_score_shift_right_padded_8<B: Backend>() {
        check_score_shift_right_padded_each::<B, 0>();
        check_score_shift_right_padded_each::<B, 1>();
        check_score_shift_right_padded_each::<B, 2>();
        check_score_shift_right_padded_each::<B, 3>();
        check_score_shift_right_padded_each::<B, 4>();
        if B::LANES >= 8 {
            check_score_shift_right_padded_each::<B, 5>();
            check_score_shift_right_padded_each::<B, 6>();
            check_score_shift_right_padded_each::<B, 7>();
            check_score_shift_right_padded_each::<B, 8>();
        }
        // L = 16 is used by the 32-lane u8 backends' propagate_horizontal_gaps.
        // Can't gate on B::LANES inside a const-generic instantiation because
        // dead branches still monomorphize. Use a separate top-level test
        // function for 32-lane coverage.
    }

    fn check_score_shift_right_padded_16<B: Backend>() {
        check_score_shift_right_padded_each::<B, 16>();
    }

    fn check_score_shift_right_padded_32<B: Backend>() {
        check_score_shift_right_padded_each::<B, 32>();
    }

    // ----- Dispatch -------------------------------------------------------

    macro_rules! backend_tests {
        ($mod_name:ident, $backend:ty) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn bytes_splat() {
                    check_bytes_splat::<$backend>();
                }
                #[test]
                fn bytes_eq() {
                    check_bytes_eq::<$backend>();
                }
                #[test]
                fn bytes_gt_lt() {
                    check_bytes_gt_lt::<$backend>();
                }
                #[test]
                fn mask_and_or_not() {
                    check_mask_and_or_not::<$backend>();
                }
                #[test]
                fn mask_zero() {
                    check_mask_zero::<$backend>();
                }
                #[test]
                fn bytes_load_partial() {
                    check_bytes_load_partial::<$backend>();
                }
                #[test]
                fn mask_shift_right_padded_1() {
                    check_mask_shift_right_padded_1::<$backend>();
                }
                #[test]
                fn score_zero() {
                    check_score_zero::<$backend>();
                }
                #[test]
                fn score_splat() {
                    check_score_splat::<$backend>();
                }
                #[test]
                fn score_first_lane() {
                    check_score_first_lane::<$backend>();
                }
                #[test]
                fn score_max_add_subs() {
                    check_score_max_add_subs::<$backend>();
                }
                #[test]
                fn score_horizontal_max() {
                    check_score_horizontal_max::<$backend>();
                }
                #[test]
                fn score_find_lane() {
                    check_score_find_lane::<$backend>();
                }
                #[test]
                fn score_shift_right_padded() {
                    check_score_shift_right_padded_8::<$backend>();
                }
            }
        };
    }

    /// Extra coverage for 32-lane backends only (uses L = 16 const).
    macro_rules! backend_tests_32_lane {
        ($mod_name:ident, $backend:ty) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn score_shift_right_padded_16() {
                    check_score_shift_right_padded_16::<$backend>();
                }
            }
        };
    }

    /// Backend tests gated on `Backend::is_available()` at runtime. Used for
    /// backends whose required CPU features may not be present in CI
    /// (notably AVX-512). Tests no-op on machines without the features
    macro_rules! backend_tests_runtime_gated {
        ($mod_name:ident, $backend:ty) => {
            mod $mod_name {
                use super::*;

                #[inline]
                fn skip_if_unavailable() -> bool {
                    !<$backend>::is_available()
                }

                #[test]
                fn bytes_splat() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_bytes_splat::<$backend>();
                }
                #[test]
                fn bytes_eq() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_bytes_eq::<$backend>();
                }
                #[test]
                fn bytes_gt_lt() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_bytes_gt_lt::<$backend>();
                }
                #[test]
                fn mask_and_or_not() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_mask_and_or_not::<$backend>();
                }
                #[test]
                fn mask_zero() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_mask_zero::<$backend>();
                }
                #[test]
                fn bytes_load_partial() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_bytes_load_partial::<$backend>();
                }
                #[test]
                fn mask_shift_right_padded_1() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_mask_shift_right_padded_1::<$backend>();
                }
                #[test]
                fn score_zero() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_zero::<$backend>();
                }
                #[test]
                fn score_splat() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_splat::<$backend>();
                }
                #[test]
                fn score_first_lane() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_first_lane::<$backend>();
                }
                #[test]
                fn score_max_add_subs() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_max_add_subs::<$backend>();
                }
                #[test]
                fn score_horizontal_max() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_horizontal_max::<$backend>();
                }
                #[test]
                fn score_find_lane() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_find_lane::<$backend>();
                }
                #[test]
                fn score_shift_right_padded() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_shift_right_padded_8::<$backend>();
                }
                #[test]
                fn score_shift_right_padded_16() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_shift_right_padded_16::<$backend>();
                }
                #[test]
                fn score_shift_right_padded_32() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_shift_right_padded_32::<$backend>();
                }
            }
        };
    }

    backend_tests!(scalar8, super::BackendScalar8);
    backend_tests!(scalar16_u8, super::BackendScalar16U8);
    #[cfg(target_arch = "x86_64")]
    backend_tests!(sse, super::BackendSSE);
    #[cfg(target_arch = "x86_64")]
    backend_tests!(sse_u8, super::BackendSSEU8);
    #[cfg(target_arch = "x86_64")]
    backend_tests!(avx, super::BackendAVX);
    #[cfg(target_arch = "x86_64")]
    backend_tests!(avx_u8, super::BackendAVXU8);
    #[cfg(target_arch = "x86_64")]
    backend_tests_32_lane!(avx_u8_extra, super::BackendAVXU8);
    #[cfg(target_arch = "x86_64")]
    backend_tests_runtime_gated!(avx512, super::BackendAVX512);
    #[cfg(target_arch = "x86_64")]
    backend_tests_runtime_gated!(avx512_u8, super::BackendAVX512U8);
    #[cfg(target_arch = "aarch64")]
    backend_tests!(neon, super::BackendNEON);
    #[cfg(target_arch = "aarch64")]
    backend_tests!(neon_u8, super::BackendNEONU8);

    // ---------------------------------------------------------------
    // Cross-backend parity: every available backend should produce the same
    // scores and the same alignment-path indices as the runtime-selected
    // backend. This covers u8 and u16 paths on each architecture.
    // ---------------------------------------------------------------

    fn cases() -> Vec<(&'static str, &'static str)> {
        vec![
            // short
            ("a", "abc"),
            ("abc", "abc"),
            ("foo", "fooBar"),
            // crossing 8-byte chunk boundary (SSE u16 LANES = 8)
            ("foo", "012345foo"),
            ("foo", "01234567foo"),
            ("foo", "0123456789foo"),
            // crossing 16-byte boundary (AVX u16, SSE u8 LANES = 16)
            ("foo", "0123456789012345foo"),
            // crossing 32-byte boundary (AVX u8 LANES = 32)
            ("foo", "0123456789012345678901234567foo"),
            // ranges that cross multiple chunks for all widths
            ("test", "Utooooeoooosoooot"),
            ("test", "Utooooooeoooooosoooooot"),
            // typos
            ("foo", "Ufooo"),
            ("foo", "Ufo"),
            // delimiter / capitalization
            ("hw", "hello_world"),
            ("fBr", "fooBar"),
            ("D", "FOR_DIST"),
            // long needles (some short enough for u8, some not)
            ("needle", "____________needle____________"),
            ("abcdefghij", "abcdefghij"),
            ("abcdefghijklmnopqrst", "abcdefghijklmnopqrst"),
        ]
    }

    fn score_with<B: Backend>(needle: &str, haystack: &str) -> u16 {
        let mut matcher = SmithWaterman::<B>::new(needle.as_bytes(), &Scoring::default(), false);
        matcher.match_haystack(haystack.as_bytes(), None).unwrap()
    }

    fn indices_with<B: Backend>(needle: &str, haystack: &str) -> Option<Vec<usize>> {
        let mut matcher = SmithWaterman::<B>::new(needle.as_bytes(), &Scoring::default(), false);
        matcher
            .match_haystack_indices(haystack.as_bytes(), 0, None)
            .map(|(_, indices)| indices)
    }

    fn assert_score_backend<B: Backend>(label: &str, needle: &str, haystack: &str, want: u16) {
        if B::is_available() {
            let got = score_with::<B>(needle, haystack);
            assert_eq!(
                got, want,
                "{label} score mismatch for needle={needle:?} haystack={haystack:?}"
            );
        }
    }

    fn assert_indices_backend<B: Backend>(
        label: &str,
        needle: &str,
        haystack: &str,
        want: Option<Vec<usize>>,
    ) {
        if B::is_available() {
            let got = indices_with::<B>(needle, haystack);
            assert_eq!(
                got, want,
                "{label} indices mismatch for needle={needle:?} haystack={haystack:?}"
            );
        }
    }

    #[test]
    fn cross_backend_parity_score() {
        for (needle, haystack) in cases() {
            let want = score_with::<BackendScalar8>(needle, haystack);

            #[cfg(target_arch = "x86_64")]
            {
                assert_score_backend::<BackendSSE>("SSE-u16", needle, haystack, want);
                assert_score_backend::<BackendAVX512>("AVX-512-u16", needle, haystack, want);
                assert_score_backend::<BackendAVX>("AVX-u16", needle, haystack, want);

                if score_fits_in_u8(needle.len(), &Scoring::default()) {
                    assert_score_backend::<BackendSSEU8>("SSE-u8", needle, haystack, want);
                    assert_score_backend::<BackendAVXU8>("AVX-u8", needle, haystack, want);
                    assert_score_backend::<BackendAVX512U8>("AVX-512-u8", needle, haystack, want);
                }
            }

            assert_score_backend::<BackendScalar8>("Scalar-u16", needle, haystack, want);

            if score_fits_in_u8(needle.len(), &Scoring::default()) {
                assert_score_backend::<BackendScalar16U8>("Scalar-u8", needle, haystack, want);
            }
        }
    }

    #[test]
    fn cross_backend_parity_indices() {
        for (needle, haystack) in cases() {
            let want = indices_with::<BackendScalar8>(needle, haystack);

            #[cfg(target_arch = "x86_64")]
            {
                assert_indices_backend::<BackendSSE>("SSE-u16", needle, haystack, want.clone());
                assert_indices_backend::<BackendAVX512>(
                    "AVX-512-u16",
                    needle,
                    haystack,
                    want.clone(),
                );
                assert_indices_backend::<BackendAVX>("AVX-u16", needle, haystack, want.clone());

                if score_fits_in_u8(needle.len(), &Scoring::default()) {
                    assert_indices_backend::<BackendSSEU8>(
                        "SSE-u8",
                        needle,
                        haystack,
                        want.clone(),
                    );
                    assert_indices_backend::<BackendAVXU8>(
                        "AVX-u8",
                        needle,
                        haystack,
                        want.clone(),
                    );
                    assert_indices_backend::<BackendAVX512U8>(
                        "AVX-512-u8",
                        needle,
                        haystack,
                        want.clone(),
                    );
                }
            }

            assert_indices_backend::<BackendScalar8>("Scalar-u16", needle, haystack, want.clone());

            if score_fits_in_u8(needle.len(), &Scoring::default()) {
                assert_indices_backend::<BackendScalar16U8>("Scalar-u8", needle, haystack, want);
            }
        }
    }

    fn ascii_byte() -> BoxedStrategy<u8> {
        prop_oneof![
            b'a'..=b'z',
            b'A'..=b'Z',
            b'0'..=b'9',
            prop::sample::select(b" /.,_-:".to_vec()),
        ]
        .boxed()
    }

    fn byte_vec(max_len: usize) -> BoxedStrategy<Vec<u8>> {
        let boundary_lengths = [0usize, 1, 7, 8, 15, 16, 31, 32, 63, 64, 511, 512, 513]
            .into_iter()
            .filter(move |len| *len <= max_len)
            .collect::<Vec<_>>();
        let regular = prop::collection::vec(ascii_byte(), 0..=max_len).boxed();
        let boundary = prop::sample::select(boundary_lengths)
            .prop_flat_map(|len| prop::collection::vec(ascii_byte(), len))
            .boxed();

        prop_oneof![4 => regular, 1 => boundary].boxed()
    }

    fn score_bytes_with<B: Backend>(needle: &[u8], haystack: &[u8], case_sensitive: bool) -> u16 {
        let mut matcher = SmithWaterman::<B>::new(needle, &Scoring::default(), case_sensitive);
        matcher.score_haystack(haystack)
    }

    fn indices_bytes_with<B: Backend>(
        needle: &[u8],
        haystack: &[u8],
        max_typos: Option<u16>,
        case_sensitive: bool,
    ) -> Option<(u16, Vec<usize>)> {
        let mut matcher = SmithWaterman::<B>::new(needle, &Scoring::default(), case_sensitive);
        matcher.match_haystack_indices(haystack, 0, max_typos)
    }

    fn prop_assert_backend<B: Backend>(
        label: &str,
        needle: &[u8],
        haystack: &[u8],
        max_typos: Option<u16>,
        case_sensitive: bool,
        want_score: u16,
        want_indices_score: Option<u16>,
    ) -> Result<(), TestCaseError> {
        if B::is_available() {
            prop_assert_eq!(
                score_bytes_with::<B>(needle, haystack, case_sensitive),
                want_score,
                "{} score mismatch",
                label
            );
            let indices = indices_bytes_with::<B>(needle, haystack, max_typos, case_sensitive);
            prop_assert_eq!(
                indices.as_ref().map(|(score, _)| *score),
                want_indices_score,
                "{} indexed score mismatch",
                label
            );
            if let Some((_, indices)) = indices {
                prop_assert_indices_valid(label, needle, haystack, &indices)?;
            }
        }
        Ok(())
    }

    fn prop_assert_backend_matches_reference<B: Backend, R: Backend>(
        label: &str,
        needle: &[u8],
        haystack: &[u8],
        max_typos: Option<u16>,
        case_sensitive: bool,
    ) -> Result<(), TestCaseError> {
        if B::is_available() {
            let want_score = score_bytes_with::<R>(needle, haystack, case_sensitive);
            let want_indices_score =
                indices_bytes_with::<R>(needle, haystack, max_typos, case_sensitive)
                    .as_ref()
                    .map(|(score, _)| *score);
            prop_assert_backend::<B>(
                label,
                needle,
                haystack,
                max_typos,
                case_sensitive,
                want_score,
                want_indices_score,
            )?;
        }
        Ok(())
    }

    fn prop_assert_indices_valid(
        label: &str,
        needle: &[u8],
        haystack: &[u8],
        indices: &[usize],
    ) -> Result<(), TestCaseError> {
        prop_assert!(
            indices.windows(2).all(|window| window[0] > window[1]),
            "{} indices are not in reverse order: {:?}",
            label,
            indices
        );
        prop_assert!(
            indices.len() <= needle.len(),
            "{} indices contain more positions than needle bytes: indices={:?} needle_len={}",
            label,
            indices,
            needle.len()
        );
        for &index in indices {
            prop_assert!(
                index < haystack.len(),
                "{} index {} is out of bounds for haystack_len={}",
                label,
                index,
                haystack.len()
            );
        }
        Ok(())
    }

    fn proptest_config(cases: u32, max_shrink_iters: u32) -> ProptestConfig {
        let mut config = ProptestConfig {
            cases,
            max_shrink_iters,
            ..ProptestConfig::default()
        };
        if cfg!(miri) {
            config.cases = cases.min(4);
            config.max_shrink_iters = max_shrink_iters.min(64);
            config.failure_persistence = None;
        }
        config
    }

    fn proptest_bound(max: usize, miri_max: usize) -> usize {
        if cfg!(miri) { max.min(miri_max) } else { max }
    }

    proptest! {
        #![proptest_config(proptest_config(192, 2048))]

        #[test]
        fn randomized_cross_backend_parity(
            needle in byte_vec(proptest_bound(96, 32)),
            haystack in byte_vec(proptest_bound(768, 128)),
            max_typos in prop_oneof![Just(None), (0u16..=16).prop_map(Some)],
            case_sensitive in any::<bool>(),
        ) {
            prop_assume!(!needle.is_empty());
            prop_assume!(!needle.contains(&0) && !haystack.contains(&0));

            prop_assert_backend_matches_reference::<BackendScalar8, BackendScalar8>(
                "scalar-u16",
                &needle,
                &haystack,
                max_typos,
                case_sensitive,
            )?;

            if score_fits_in_u8(needle.len(), &Scoring::default()) {
                prop_assert_backend_matches_reference::<BackendScalar16U8, BackendScalar16U8>(
                    "scalar-u8",
                    &needle,
                    &haystack,
                    max_typos,
                    case_sensitive,
                )?;
            }

            #[cfg(target_arch = "x86_64")]
            {
                prop_assert_backend_matches_reference::<BackendSSE, BackendScalar8>(
                    "SSE-u16",
                    &needle,
                    &haystack,
                    max_typos,
                    case_sensitive,
                )?;
                prop_assert_backend_matches_reference::<BackendAVX, TestScalar16>(
                    "AVX-u16",
                    &needle,
                    &haystack,
                    max_typos,
                    case_sensitive,
                )?;
                prop_assert_backend_matches_reference::<BackendAVX512, TestScalar32>(
                    "AVX-512-u16",
                    &needle,
                    &haystack,
                    max_typos,
                    case_sensitive,
                )?;

                if score_fits_in_u8(needle.len(), &Scoring::default()) {
                    prop_assert_backend_matches_reference::<BackendSSEU8, BackendScalar16U8>(
                        "SSE-u8",
                        &needle,
                        &haystack,
                        max_typos,
                        case_sensitive,
                    )?;
                    prop_assert_backend_matches_reference::<BackendAVXU8, TestScalar32U8>(
                        "AVX-u8",
                        &needle,
                        &haystack,
                        max_typos,
                        case_sensitive,
                    )?;
                    prop_assert_backend_matches_reference::<BackendAVX512U8, TestScalar64U8>(
                        "AVX-512-u8",
                        &needle,
                        &haystack,
                        max_typos,
                        case_sensitive,
                    )?;
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                prop_assert_backend_matches_reference::<BackendNEON, BackendScalar8>(
                    "NEON-u16",
                    &needle,
                    &haystack,
                    max_typos,
                    case_sensitive,
                )?;

                if score_fits_in_u8(needle.len(), &Scoring::default()) {
                    prop_assert_backend_matches_reference::<BackendNEONU8, BackendScalar16U8>(
                        "NEON-u8",
                        &needle,
                        &haystack,
                        max_typos,
                        case_sensitive,
                    )?;
                }
            }
        }
    }
}
