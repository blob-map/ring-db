use bytemuck::{Pod, Zeroable};
use ringdb::{
    CandidateSet, DiskIntersectionQuery, DiskQuery, KnnQuery, MetadataKey, MetadataTag,
    MetadataValue, RangeQuery, RingDb, RingDbConfig, RingDbError, RingQuery,
};

const KIND: MetadataKey = MetadataKey::new(1);
const STATUS: MetadataKey = MetadataKey::new(2);
const BAKERY: MetadataValue = MetadataValue::new(10);
const CAFE: MetadataValue = MetadataValue::new(11);
const PHARMACY: MetadataValue = MetadataValue::new(12);
const OPEN: MetadataValue = MetadataValue::new(20);
const CLOSED: MetadataValue = MetadataValue::new(21);

const BAKERY_TAG: MetadataTag = MetadataTag::new(KIND, BAKERY);
const CAFE_TAG: MetadataTag = MetadataTag::new(KIND, CAFE);
const PHARMACY_TAG: MetadataTag = MetadataTag::new(KIND, PHARMACY);
const OPEN_TAG: MetadataTag = MetadataTag::new(STATUS, OPEN);
const CLOSED_TAG: MetadataTag = MetadataTag::new(STATUS, CLOSED);

fn set_ids(set: &CandidateSet) -> Vec<u32> {
    (0..set.universe_len() as u32)
        .filter(|&id| set.contains(id).unwrap())
        .collect()
}

fn sorted_result_ids(mut ids: Vec<u32>) -> Vec<u32> {
    ids.sort_unstable();
    ids
}

fn tagged_places() -> ringdb::SealedRingDb {
    let rows = [
        ([1.0, 0.0], [BAKERY_TAG, OPEN_TAG]),
        ([2.0, 0.0], [CAFE_TAG, OPEN_TAG]),
        ([3.0, 0.0], [BAKERY_TAG, CLOSED_TAG]),
        ([4.0, 0.0], [CAFE_TAG, CLOSED_TAG]),
        ([5.0, 0.0], [BAKERY_TAG, OPEN_TAG]),
        ([6.0, 0.0], [PHARMACY_TAG, OPEN_TAG]),
    ];
    let mut db = RingDb::new(RingDbConfig::new(2)).unwrap();
    for (vector, tags) in rows {
        db.add_vector_with_metadata(&vector, (), &tags).unwrap();
    }
    db.build().unwrap()
}

fn database_with_len(len: usize) -> ringdb::SealedRingDb {
    let mut db = RingDb::new(RingDbConfig::new(1)).unwrap();
    for id in 0..len {
        db.add_vector(&[id as f32], ()).unwrap();
    }
    db.build().unwrap()
}

#[test]
fn explicit_metadata_tags_build_reusable_candidate_sets() {
    let mut builder = RingDb::new(RingDbConfig::new(2)).unwrap();
    builder
        .add_vector_with_metadata(&[1.0, 0.0], (), &[BAKERY_TAG, BAKERY_TAG, OPEN_TAG])
        .unwrap();
    builder.add_vector(&[2.0, 0.0], ()).unwrap();
    builder
        .add_vector_with_metadata(&[3.0, 0.0], (), &[BAKERY_TAG, CLOSED_TAG])
        .unwrap();
    let db = builder.build().unwrap();

    let bakeries = db.metadata_eq(KIND, BAKERY);
    assert_eq!(bakeries.universe_len(), 3);
    assert_eq!(set_ids(&bakeries), [0, 2]);
    assert_eq!(bakeries.len(), 2, "duplicate tags must not duplicate IDs");

    let missing = db.metadata_eq(MetadataKey::new(999), MetadataValue::new(999));
    assert_eq!(missing.universe_len(), 3);
    assert!(missing.is_empty());
}

#[test]
fn candidate_set_operations_are_validated_and_deterministic() {
    let db = database_with_len(8);
    let left = db.candidates_from_ids([7, 3, 1, 3]).unwrap();
    let right = db.candidates_from_ids([6, 3, 2]).unwrap();

    assert_eq!(set_ids(&left), [1, 3, 7]);
    assert_eq!(set_ids(&left.intersection(&right).unwrap()), [3]);
    assert_eq!(set_ids(&left.union(&right).unwrap()), [1, 2, 3, 6, 7]);
    assert_eq!(set_ids(&left.complement()), [0, 2, 4, 5, 6]);
    assert_eq!(
        set_ids(&left.union(&db.candidates_from_ids(0..8).unwrap()).unwrap()),
        (0..8).collect::<Vec<_>>()
    );
    assert!(
        left.intersection(&db.candidates_from_ids([]).unwrap())
            .unwrap()
            .is_empty()
    );

    assert!(matches!(
        db.candidates_from_ids([8]),
        Err(RingDbError::InvalidVectorId { id: 8, len: 8 })
    ));
    assert!(matches!(
        left.contains(8),
        Err(RingDbError::InvalidVectorId { id: 8, len: 8 })
    ));
    assert!(matches!(
        left.union(&database_with_len(9).candidates_from_ids(0..9).unwrap()),
        Err(RingDbError::CandidateSetMismatch {
            expected: 8,
            got: 9
        })
    ));
}

#[test]
fn metadata_candidates_filter_every_geometric_query_shape() {
    let db = tagged_places();
    let origin = [0.0, 0.0];
    let center = [4.0, 0.0];
    let bakeries = db.metadata_eq(KIND, BAKERY);
    let open = db.metadata_eq(STATUS, OPEN);
    let open_bakeries = bakeries.intersection(&open).unwrap();

    let ring = db
        .query(&RingQuery::new(&origin, 3.0, 2.0).with_candidates(&open_bakeries))
        .unwrap();
    assert_eq!(sorted_result_ids(ring.ids()), [0, 4]);

    let range = db
        .query_range(&RangeQuery::new(&origin, 2.0, 5.0).with_candidates(&bakeries))
        .unwrap();
    assert_eq!(sorted_result_ids(range.ids()), [2, 4]);

    let disk = db
        .query_disk(&DiskQuery::new(&origin, 4.0).with_candidates(&bakeries))
        .unwrap();
    assert_eq!(sorted_result_ids(disk.ids()), [0, 2]);

    let disks = [
        DiskQuery::new(&origin, 6.0).with_candidates(&open),
        DiskQuery::new(&center, 2.0),
    ];
    let intersection = db
        .query_disk_intersection(&DiskIntersectionQuery::new(&disks).with_candidates(&bakeries))
        .unwrap();
    assert_eq!(sorted_result_ids(intersection.ids()), [4]);
}

#[test]
fn exact_knn_is_weighted_filterable_and_deterministic() {
    let vectors = [
        ([1.0, 0.0], CAFE_TAG),
        ([-1.0, 0.0], CAFE_TAG),
        ([0.0, 2.0], BAKERY_TAG),
        ([3.0, 0.0], BAKERY_TAG),
        ([0.0, 4.0], BAKERY_TAG),
    ];
    let mut builder = RingDb::new(RingDbConfig::new(2)).unwrap();
    for (vector, tag) in vectors {
        builder
            .add_vector_with_metadata(&vector, (), &[tag])
            .unwrap();
    }
    let db = builder.build().unwrap();
    let origin = [0.0, 0.0];

    let nearest = db.query_knn(&KnnQuery::new(&origin, 3)).unwrap();
    assert_eq!(nearest.ids(), [0, 1, 2]);
    assert_eq!(
        nearest
            .hits
            .iter()
            .map(|hit| hit.dist_sq)
            .collect::<Vec<_>>(),
        [1.0, 1.0, 4.0]
    );

    let weights = [0.25, 4.0];
    let weighted = db
        .query_knn(&KnnQuery::new(&origin, 3).with_weights(&weights))
        .unwrap();
    assert_eq!(weighted.ids(), [0, 1, 3]);
    assert_eq!(
        weighted
            .hits
            .iter()
            .map(|hit| hit.dist_sq)
            .collect::<Vec<_>>(),
        [0.25, 0.25, 2.25]
    );

    let bakeries = db.metadata_eq(KIND, BAKERY);
    let filtered = db
        .query_knn(
            &KnnQuery::new(&origin, 2)
                .with_weights(&[4.0, 0.25])
                .with_candidates(&bakeries),
        )
        .unwrap();
    assert_eq!(filtered.ids(), [2, 4]);
    assert_eq!(
        filtered
            .hits
            .iter()
            .map(|hit| hit.dist_sq)
            .collect::<Vec<_>>(),
        [1.0, 4.0]
    );
}

#[test]
fn unweighted_knn_preserves_small_distances_at_large_coordinates() {
    let query = [10_000_000.0f32];
    let mut builder = RingDb::new(RingDbConfig::new(1)).unwrap();
    for vector in [[10_000_003.0], [10_000_001.0], [10_000_002.0]] {
        builder.add_vector(&vector, ()).unwrap();
    }
    let db = builder.build().unwrap();

    let result = db.query_knn(&KnnQuery::new(&query, 3)).unwrap();
    assert_eq!(result.ids(), [1, 2, 0]);
    assert_eq!(
        result
            .hits
            .iter()
            .map(|hit| hit.dist_sq)
            .collect::<Vec<_>>(),
        [1.0, 4.0, 9.0]
    );
}

#[test]
fn zero_weight_ignores_even_overflowing_coordinate_differences() {
    let mut builder = RingDb::new(RingDbConfig::new(2)).unwrap();
    builder.add_vector(&[-f32::MAX, 0.0], ()).unwrap();
    builder.add_vector(&[0.0, 1.0], ()).unwrap();
    builder.add_vector(&[f32::MAX, 0.0], ()).unwrap();
    let db = builder.build().unwrap();

    let query = [f32::MAX, 0.0];
    let weights = [0.0, 1.0];
    let result = db
        .query_knn(&KnnQuery::new(&query, 3).with_weights(&weights))
        .unwrap();

    assert_eq!(result.ids(), [0, 2, 1]);
    assert_eq!(
        result
            .hits
            .iter()
            .map(|hit| hit.dist_sq)
            .collect::<Vec<_>>(),
        [0.0, 0.0, 1.0]
    );
}

#[test]
fn exact_knn_handles_zero_large_and_empty_candidate_sets() {
    let db = tagged_places();
    let origin = [0.0, 0.0];

    assert!(
        db.query_knn(&KnnQuery::new(&origin, 0))
            .unwrap()
            .hits
            .is_empty()
    );

    let all = db.query_knn(&KnnQuery::new(&origin, 100)).unwrap();
    assert_eq!(all.ids(), [0, 1, 2, 3, 4, 5]);

    let bakeries = db.metadata_eq(KIND, BAKERY);
    let filtered = db
        .query_knn(&KnnQuery::new(&origin, 100).with_candidates(&bakeries))
        .unwrap();
    assert_eq!(filtered.ids(), [0, 2, 4]);

    let empty = db.metadata_eq(KIND, MetadataValue::new(999));
    assert!(
        db.query_knn(&KnnQuery::new(&origin, 10).with_candidates(&empty))
            .unwrap()
            .hits
            .is_empty()
    );

    let empty_db: ringdb::SealedRingDb =
        RingDb::new(RingDbConfig::new(2)).unwrap().build().unwrap();
    assert!(
        empty_db
            .query_knn(&KnnQuery::new(&origin, 10))
            .unwrap()
            .hits
            .is_empty()
    );
}

#[test]
fn candidate_sets_with_another_universe_are_rejected_by_queries() {
    let db = tagged_places();
    let foreign_db = database_with_len(db.len() + 1);
    let foreign = foreign_db
        .candidates_from_ids(0..foreign_db.len() as u32)
        .unwrap();
    let origin = [0.0, 0.0];

    let error = match db.query_knn(&KnnQuery::new(&origin, 1).with_candidates(&foreign)) {
        Ok(_) => panic!("a mismatched candidate universe should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        RingDbError::CandidateSetMismatch {
            expected: 6,
            got: 7
        }
    ));
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
struct PlacePayload {
    name: [u8; 8],
    rating: u32,
}

fn persist_places(path: &std::path::Path) {
    let mut db: RingDb<PlacePayload> =
        RingDb::new(RingDbConfig::new(2).with_persist_dir(path)).unwrap();
    let rows = [
        ([3.0, 0.0], *b"far\0\0\0\0\0", 30, [BAKERY_TAG, OPEN_TAG]),
        ([2.0, 0.0], *b"cafe\0\0\0\0", 40, [CAFE_TAG, OPEN_TAG]),
        ([1.0, 0.0], *b"near\0\0\0\0", 50, [BAKERY_TAG, CLOSED_TAG]),
    ];
    for (vector, name, rating, tags) in rows {
        db.add_vector_with_metadata(&vector, PlacePayload { name, rating }, &tags)
            .unwrap();
    }
    db.build().unwrap();
}

#[test]
fn metadata_and_pod_payloads_survive_reload_deterministically() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    persist_places(first.path());
    persist_places(second.path());

    let first_metadata = std::fs::read(first.path().join("metadata.bin")).unwrap();
    let second_metadata = std::fs::read(second.path().join("metadata.bin")).unwrap();
    assert_eq!(first_metadata, second_metadata);

    let loaded =
        RingDb::<PlacePayload>::load(first.path(), ringdb::BackendPreference::Cpu).unwrap();
    let bakeries = loaded.metadata_eq(KIND, BAKERY);
    assert_eq!(set_ids(&bakeries), [0, 2]);

    let nearest_bakery = loaded
        .query_knn(&KnnQuery::new(&[0.0, 0.0], 1).with_candidates(&bakeries))
        .unwrap();
    assert_eq!(nearest_bakery.ids(), [2]);
    let payload = loaded.fetch_payload(2).unwrap();
    assert_eq!(&payload.name[..4], b"near");
    assert_eq!(payload.rating, 50);
}
