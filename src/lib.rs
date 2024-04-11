use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub score: Score,
    pub doc: DocId,
}

impl PartialEq for Hit {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc == other.doc
    }
}

impl Eq for Hit {}

impl PartialOrd for Hit {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hit {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score).reverse()
    }
}

pub type DocId = u32;
pub type Score = u64;

pub trait TopK: Sized {
    fn new(k: usize) -> Self;
    fn top_k(&mut self, hits: impl Iterator<Item=Hit>, output: &mut Vec<Hit>);
}

pub struct HeapTopK {
    top_k: BinaryHeap<Hit>,
    k: usize,
}

impl TopK for HeapTopK {
    fn new(k: usize) -> Self {
        HeapTopK {
            top_k: BinaryHeap::with_capacity(k),
            k,
        }
    }

    fn top_k(&mut self, mut hits: impl Iterator<Item = Hit>, output: &mut Vec<Hit>) {
        self.top_k.clear();
        self.top_k.extend((&mut hits).take(self.k));
        let mut threshold = self.top_k.peek().unwrap().score;

        for hit in hits {
            if hit.score <= threshold {
                continue;
            }
            let mut head = self.top_k.peek_mut().unwrap();
            *head = hit;
            drop(head);
            threshold = self.top_k.peek().unwrap().score;
        }
        output.clear();
        output.reserve(self.k);
        output.extend(self.top_k.drain());
        assert_eq!(output.len(), self.k);
    }
}


pub struct MedianTopK {
    top_k: Box<[Hit]>,
    k: usize,
}

impl TopK for MedianTopK {
    fn new(k: usize) -> Self {
        let top_k = vec![Hit {score: 0, doc: 0}; 2*k].into_boxed_slice();
        MedianTopK {
            top_k,
            k,
        }
    }

    fn top_k(&mut self, mut hits: impl Iterator<Item=Hit>, output: &mut Vec<Hit>) {
        let mut threshold = 0u64;

        let mut len = self.top_k.len().min(self.k);
        for (buffer_cell, hit) in self.top_k[..len].iter_mut().zip(&mut hits) {
            *buffer_cell = hit;
        }
        for hit in hits {
            if hit.score <= threshold {
                continue;
            }
            self.top_k[len] = hit;
            len += 1;
            if len == self.top_k.len() {
                let (_, median_el, _) = self.top_k.select_nth_unstable(self.k - 1);
                threshold = median_el.score;
                len = self.k;
            }
        }

        if self.top_k.len() > self.k {
            self.top_k.select_nth_unstable(self.k - 1);
        }
        output.clear();
        output.reserve(self.k);
        output.extend(self.top_k[..self.k].iter().copied());
    }
}


pub fn simplified_heap_top_k(mut hits: impl Iterator<Item=Hit>, k: usize) -> Vec<Hit> {
    let mut top_k = BinaryHeap::with_capacity(k);
    top_k.extend((&mut hits).take(k));
    let mut threshold = top_k.peek().unwrap().score;

    for hit in hits {
        if hit.score <= threshold {
            continue;
        }
        let mut head = top_k.peek_mut().unwrap();
        *head = hit;
        drop(head);
        threshold = top_k.peek().unwrap().score;
    }

    top_k.into_sorted_vec()
}

pub fn simplified_median_top_k(mut hits: impl Iterator<Item=Hit>, k: usize) -> Vec<Hit> {
    assert!(k>0);
    let mut top_k = Vec::with_capacity(2 * k);

    // We need to prefill the array to deal with the case where
    // where we have a lot elements with a score of 0.
    top_k.extend((&mut hits).take(k));

    let mut threshold = 0u64;
    for hit in hits {
        if hit.score <= threshold {
            continue;
        }
        top_k.push(hit);
        if top_k.len() == 2*k {
            let (_, median_el, _) = top_k.select_nth_unstable(k - 1);
            threshold = median_el.score;
            top_k.truncate(k);
        }
    }
    top_k.sort_unstable();
    top_k.truncate(k);
    top_k
}

#[cfg(test)]
mod tests {

    use super::*;
    use rand;
    use rand::seq::SliceRandom;


    fn test_topk_util<T: TopK>(n: usize, k: usize) {
        let mut rng = rand::thread_rng();
        let mut hits: Vec<Hit> = (0..n).map(|i| Hit { score: i as u64, doc: i as u32 }).collect();
        let mut output = Vec::new();
        let mut top_k = T::new(k);
        for _ in 0..10 {
            hits.shuffle(&mut rng);
            top_k.top_k(hits.iter().copied(), &mut output);
            output.sort_unstable();
            let m = n.min(k);
            assert_eq!(output.len(), m);
            for (hit, expected_score) in output.iter().zip((0..n).rev()) {
                assert_eq!(hit.score, expected_score as u64);
            }
        }
    }

    #[test]
    fn test_top_k_heap() {
        test_topk_util::<HeapTopK>(100, 3);
    }

    #[test]
    fn test_top_k_fast() {
        test_topk_util::<MedianTopK>(100, 3);
    }


    #[test]
    fn test_heap_top_k_simplified() {
        let mut rng = rand::thread_rng();
        let n = 100;
        let k = 10;
        let mut hits: Vec<Hit> = (0..n).map(|i| Hit { score: i as u64, doc: i as u32 }).collect();
        for _ in 0..10 {
            hits.shuffle(&mut rng);
            let output = super::simplified_heap_top_k(hits.iter().copied(), k);
            assert_eq!(output.len(), k);
            for (hit, expected_score) in output.iter().zip((0..n).rev()) {
                assert_eq!(hit.score, expected_score as u64);
            }
        }
    }

    #[test]
    fn test_median_top_k() {
        let mut rng = rand::thread_rng();
        let n = 100;
        let k = 10;
        let mut hits: Vec<Hit> = (0..n).map(|i| Hit { score: i as u64, doc: i as u32 }).collect();
        for _ in 0..10 {
            hits.shuffle(&mut rng);
            let output = super::simplified_median_top_k(hits.iter().copied(), k);
            assert_eq!(output.len(), k);
            for (hit, expected_score) in output.iter().zip((0..n).rev()) {
                assert_eq!(hit.score, expected_score as u64);
            }
        }
    }
}
