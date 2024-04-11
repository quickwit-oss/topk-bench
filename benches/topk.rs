use topk::MedianTopK;
use topk::HeapTopK;
use topk::Hit;
use topk::TopK;
use rand::seq::SliceRandom;

fn main() {
    divan::main();
}

const KS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

#[divan::bench(types = [
    MedianTopK,
    HeapTopK,
], args = KS, max_time=1)]
fn shuffled<T>(bencher: divan::Bencher, k: usize) where T: TopK {
    bencher.with_inputs(|| {
        let n = 1_000_000;
        let mut rng = rand::thread_rng();
        let mut hits: Vec<Hit> = (0..n).map(|i| Hit { score: i as u64, doc: i as u32 }).collect();
        hits.shuffle(&mut rng);
        (T::new(k), hits, Vec::with_capacity(k))
    })
    .bench_refs(|(top_k, hits, output)| {
        top_k.top_k(hits.iter().copied(), output);
    });
}

#[divan::bench(types = [
    MedianTopK,
    HeapTopK,
], args = KS)]
fn sorted<T>(bencher: divan::Bencher, k: usize) where T: TopK {
    bencher.with_inputs(|| {
        let n = 1_000_000;
        let hits: Vec<Hit> = (0..n).map(|i| Hit { score: i as u64, doc: i as u32 }).collect();
        (T::new(k), hits, Vec::with_capacity(k))
    })
    .bench_refs(|(top_k, hits, output)| {
        top_k.top_k(hits.iter().copied(), output);
    });
}

// #[divan::bench(types = [
//     MedianTopK,
//     HeapTopK,
// ], args = KS, max_time=1)]
// fn bench_top_k_inv_sorted<T>(bencher: divan::Bencher, k: usize) where T: TopK {
//     bencher.with_inputs(|| {
//         let n = 1_000_000;
//         let mut hits: Vec<Hit> = (0..n).map(|i| Hit { score: i as u64, doc: i as u32 }).collect();
//         hits.reverse();
//         (T::new(k), Vec::with_capacity(k), hits)
//     })
//     .bench_refs(|(top_k, output, hits)| {
//         top_k.top_k(hits.iter().copied(), output);
//     });
// }

