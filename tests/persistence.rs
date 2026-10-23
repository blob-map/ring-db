//! Format v1 persistence of vectors, Pod payloads, and metadata.
mod common;

use bytemuck::{Pod, Zeroable};
use common::{GROUP, knn_keys, region_keys};
use ringdb::{
    BackendPreference, Disk, Knn, MetadataTag, MetadataValue, RingDb, RingDbConfig, RingDbError,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
struct Meta {
    label: [u8; 8],
    value: i32,
}

#[test]
fn vectors_payloads_and_metadata_survive_reload() {
    let dir = tempfile::tempdir().unwrap();
    let mut db: RingDb<Meta> =
        RingDb::new(RingDbConfig::new(2).with_persist_dir(dir.path())).unwrap();
    for (i, (vector, label)) in [
        ([1.0, 0.0], b"dog"),
        ([0.0, 5.0], b"cat"),
        ([3.0, 4.0], b"emu"),
    ]
    .into_iter()
    .enumerate()
    {
        let mut name = [0; 8];
        name[..3].copy_from_slice(label);
        let tag = MetadataTag::new(GROUP, MetadataValue::new((i % 2) as u64));
        db.add_vector_with_metadata(
            &vector,
            Meta {
                label: name,
                value: i as i32,
            },
            &[tag],
        )
        .unwrap();
    }
    let original = db.build().unwrap();
    let origin = [0.0, 0.0];
    let before = region_keys(&original.search(Disk::new(&origin, 5.0)).unwrap());
    drop(original);

    let loaded = RingDb::<Meta>::load(dir.path(), BackendPreference::Cpu).unwrap();
    assert_eq!((loaded.len(), loaded.dims()), (3, 2));
    assert_eq!(
        region_keys(&loaded.search(Disk::new(&origin, 5.0)).unwrap()),
        before
    );
    let even = loaded.metadata_eq(GROUP, MetadataValue::new(0));
    let hits = loaded.search(Knn::new(&origin, 5).filter(&even)).unwrap();
    assert_eq!(knn_keys(&hits), [0, 2]);
    let payloads = loaded
        .fetch_payloads(&hits.iter().map(|h| h.key).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(&payloads[0].label[..3], b"dog");
    assert_eq!(payloads[1].value, 2);
}

#[test]
fn persistence_refuses_to_overwrite_a_nonempty_directory() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("keep-me");
    std::fs::write(&marker, b"user data").unwrap();
    let mut db = RingDb::new(RingDbConfig::new(1).with_persist_dir(dir.path())).unwrap();
    db.add_vector(&[0.0], ()).unwrap();
    assert!(matches!(
        db.build(),
        Err(RingDbError::Io(ref error)) if error.kind() == std::io::ErrorKind::AlreadyExists
    ));
    assert_eq!(std::fs::read(marker).unwrap(), b"user data");
}

#[test]
fn load_rejects_corrupt_vectors_row_counts_and_payload_layouts() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = RingDb::new(RingDbConfig::new(1).with_persist_dir(dir.path())).unwrap();
    db.add_vector(&[1.0], 7u32).unwrap();
    db.build().unwrap();
    assert!(matches!(
        RingDb::<u64>::load(dir.path(), BackendPreference::Cpu),
        Err(RingDbError::Corrupt(_))
    ));

    let meta_path = dir.path().join("meta.bin");
    let meta = std::fs::read(&meta_path).unwrap();
    let mut too_many = meta.clone();
    too_many[24..32].copy_from_slice(&(u64::from(u32::MAX) + 1).to_le_bytes());
    std::fs::write(&meta_path, too_many).unwrap();
    assert!(matches!(
        RingDb::<u32>::load(dir.path(), BackendPreference::Cpu),
        Err(RingDbError::Corrupt(_))
    ));

    std::fs::write(&meta_path, meta).unwrap();
    std::fs::write(dir.path().join("vectors.bin"), f32::NAN.to_le_bytes()).unwrap();
    assert!(matches!(
        RingDb::<u32>::load(dir.path(), BackendPreference::Cpu),
        Err(RingDbError::Corrupt(_))
    ));
}
