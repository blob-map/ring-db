//! Randomized searches compared with a scalar f64 reference.
mod common;

use common::{cpu, knn_keys, reference_dist};
use rand::{RngExt, SeedableRng};
use ringdb::{CandidateSet, Disk, Knn, Metric, Region, SealedRingDb};

struct Case {
    vectors: Vec<Vec<f32>>,
    db: SealedRingDb<u32>,
}

fn random_vectors(rng: &mut impl RngExt, n: usize, dims: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dims).map(|_| rng.random_range(-3.0f32..3.0)).collect())
        .collect()
}

fn random_weights(rng: &mut impl RngExt, dims: usize) -> Vec<f32> {
    (0..dims)
        .map(|i| {
            if i % 7 == 3 {
                0.0
            } else {
                rng.random_range(0.05f32..3.0)
            }
        })
        .collect()
}

/// Filters covering all, sparse, dense, and complemented representations.
fn filters(db: &SealedRingDb<u32>) -> Vec<(Option<CandidateSet>, Vec<bool>)> {
    let n = db.len() as u32;
    let sets = [
        db.candidates_from_ids(0..n).unwrap(),
        db.candidates_from_ids([0, 5, n / 2, n - 1]).unwrap(),
        db.candidates_from_ids((0..n).filter(|i| i % 3 != 1))
            .unwrap(),
        db.candidates_from_ids([1, 2, 64]).unwrap().complement(),
    ];
    std::iter::once((None, vec![true; n as usize]))
        .chain(sets.into_iter().map(|set| {
            let members = (0..n).map(|id| set.contains(id).unwrap()).collect();
            (Some(set), members)
        }))
        .collect()
}

/// A radius halfway between two reference distances, so f32 rounding cannot
/// move a row across the boundary.
fn split_radius(mut distances: Vec<f64>, fraction: f64) -> f32 {
    distances.sort_by(f64::total_cmp);
    let split = ((distances.len() as f64 * fraction) as usize).clamp(1, distances.len() - 1);
    ((distances[split - 1].sqrt() + distances[split].sqrt()) * 0.5) as f32
}

fn check_knn(
    case: &Case,
    query: &[f32],
    weights: Option<&[f32]>,
    members: &[bool],
    hits: &[ringdb::Hit],
    k: usize,
) {
    let mut expected: Vec<(f64, u64)> = case
        .vectors
        .iter()
        .enumerate()
        .filter(|(id, _)| members[*id])
        .map(|(id, v)| (reference_dist(v, query, weights), id as u64))
        .collect();
    expected.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    expected.truncate(k);
    assert_eq!(hits.len(), expected.len());
    for (hit, (dist, key)) in hits.iter().zip(&expected) {
        // Ties within rounding may swap neighbors; distances must agree.
        common::assert_close(hit.dist_sq, *dist);
        if hit.key.get() != *key {
            let actual = reference_dist(&case.vectors[hit.key.get() as usize], query, weights);
            common::assert_close(hit.dist_sq, actual);
        }
    }
}

#[test]
fn knn_and_regions_match_the_reference_for_every_filter_and_metric() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0x5EED_0002);
    for dims in [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 64, 65, 128] {
        let vectors = random_vectors(&mut rng, 131, dims);
        let case = Case {
            db: cpu(&vectors),
            vectors,
        };
        let centers = random_vectors(&mut rng, 3, dims);
        let weights = [
            random_weights(&mut rng, dims),
            random_weights(&mut rng, dims),
        ];
        let metrics: [Option<&[f32]>; 3] = [None, Some(&weights[0]), Some(&weights[1])];
        let radii: Vec<f32> = (0..3)
            .map(|i| {
                let distances = case
                    .vectors
                    .iter()
                    .map(|v| reference_dist(v, &centers[i], metrics[i]))
                    .collect();
                split_radius(distances, 0.6)
            })
            .collect();
        let disks: Vec<Disk<'_>> = (0..3)
            .map(|i| {
                let disk = Disk::new(&centers[i], radii[i]);
                match metrics[i] {
                    Some(w) => disk.metric(Metric::WeightedL2(w)),
                    None => disk,
                }
            })
            .collect();
        for (filter, members) in filters(&case.db) {
            for weights in metrics {
                for k in [0, 1, 7, 200] {
                    let mut query = Knn::new(&centers[0], k);
                    if let Some(w) = weights {
                        query = query.metric(Metric::WeightedL2(w));
                    }
                    if let Some(filter) = &filter {
                        query = query.filter(filter);
                    }
                    let hits = case.db.search(query).unwrap();
                    check_knn(&case, &centers[0], weights, &members, &hits, k);
                    if k == 7 {
                        assert_eq!(
                            knn_keys(&hits).len(),
                            7.min(members.iter().filter(|m| **m).count())
                        );
                    }
                }
            }
            for n_disks in 1..=3 {
                let mut region = Region::from(disks[0]);
                for disk in &disks[1..n_disks] {
                    region = region.and(*disk);
                }
                if let Some(filter) = &filter {
                    region = region.filter(filter);
                }
                let result = case.db.search(region).unwrap();
                let expected: Vec<(u64, Vec<f64>)> = case
                    .vectors
                    .iter()
                    .enumerate()
                    .filter(|(id, _)| members[*id])
                    .filter_map(|(id, v)| {
                        let dists: Vec<f64> = (0..n_disks)
                            .map(|i| reference_dist(v, &centers[i], metrics[i]))
                            .collect();
                        let inside =
                            (0..n_disks).all(|i| dists[i] <= f64::from(radii[i] * radii[i]));
                        inside.then_some((id as u64, dists))
                    })
                    .collect();
                let actual = common::region_rows(&result);
                assert_eq!(
                    actual.iter().map(|r| r.0).collect::<Vec<_>>(),
                    expected.iter().map(|r| r.0).collect::<Vec<_>>(),
                    "dims={dims}, disks={n_disks}"
                );
                for ((_, actual), (_, expected)) in actual.iter().zip(&expected) {
                    assert_eq!(actual.len(), n_disks);
                    for (a, e) in actual.iter().zip(expected) {
                        common::assert_close(*a, *e);
                    }
                }
            }
        }
    }
}
