use crate::smith_waterman::{AlignmentPathIter, SmithWaterman};

use super::backend::Backend;

impl<B: Backend> SmithWaterman<B> {
    #[inline(always)]
    pub fn iter_alignment_path<'a>(
        &'a self,
        needle_len: usize,
        skipped_chars: usize,
        unicode_haystack: Option<&'a [u8]>,
        score: u16,
        max_typos: Option<u16>,
    ) -> AlignmentPathIter<'a> {
        AlignmentPathIter::new::<B>(
            &self.score_matrix,
            &self.match_masks,
            needle_len,
            self.haystack_chunks,
            skipped_chars,
            unicode_haystack,
            score,
            max_typos,
        )
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn has_alignment_path(&self, score: u16, max_typos: u16) -> bool {
        let iter = self.iter_alignment_path(self.needle.len(), 0, None, score, Some(max_typos));
        for pos in iter {
            if pos.is_none() {
                return false;
            }
        }
        true
    }
}
