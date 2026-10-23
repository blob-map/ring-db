//! Field values, records to add, and documents read back.
use std::sync::Arc;

use crate::error::{Result, RingDbError};
use crate::query::Key;
use crate::schema::{FieldType, Schema};

/// The value of one field.
///
/// Integer values convert to any integer field type that can hold them, so
/// `record.set("rooms", 3)` works for a `U32` field.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U32(u32),
    U64(u64),
    I64(i64),
    F32(f32),
    Bool(bool),
    /// A `Keyword` or `Text` value.
    Text(String),
    /// A `Keywords` or `TextList` value.
    TextList(Vec<String>),
    Bytes(Vec<u8>),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        self.as_int().and_then(|v| v.try_into().ok())
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_int().and_then(|v| v.try_into().ok())
    }

    pub fn as_i64(&self) -> Option<i64> {
        self.as_int().and_then(|v| v.try_into().ok())
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_strs(&self) -> Option<&[String]> {
        match self {
            Self::TextList(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(v) => Some(v),
            _ => None,
        }
    }

    pub(crate) fn as_int(&self) -> Option<i128> {
        match *self {
            Self::U32(v) => Some(v.into()),
            Self::U64(v) => Some(v.into()),
            Self::I64(v) => Some(v.into()),
            _ => None,
        }
    }

    /// Convert to the canonical variant of `ty`, or `None` if it does not fit.
    pub(crate) fn coerce(self, ty: FieldType) -> Option<Self> {
        let int = self.as_int();
        match (ty, self) {
            (FieldType::U32, _) => int?.try_into().ok().map(Self::U32),
            (FieldType::U64, _) => int?.try_into().ok().map(Self::U64),
            (FieldType::I64, _) => int?.try_into().ok().map(Self::I64),
            (FieldType::F32, Self::F32(v)) => v.is_finite().then_some(Self::F32(v)),
            (FieldType::Bool, v @ Self::Bool(_))
            | (FieldType::Keyword | FieldType::Text, v @ Self::Text(_))
            | (FieldType::Keywords | FieldType::TextList, v @ Self::TextList(_))
            | (FieldType::Bytes, v @ Self::Bytes(_)) => Some(v),
            (FieldType::Keywords | FieldType::TextList, Self::Text(v)) => {
                Some(Self::TextList(vec![v]))
            }
            _ => None,
        }
    }

    /// Bytes this value adds to a stored record.
    pub(crate) fn stored_len(&self) -> usize {
        match self {
            Self::U32(_) | Self::F32(_) => 4,
            Self::U64(_) | Self::I64(_) => 8,
            Self::Bool(_) => 1,
            Self::Text(v) => v.len(),
            Self::TextList(v) => v.iter().map(|s| s.len() + 4).sum(),
            Self::Bytes(v) => v.len(),
        }
    }
}

macro_rules! from_int {
    ($($t:ty => $variant:ident),*) => {$(
        impl From<$t> for Value {
            fn from(value: $t) -> Self {
                Self::$variant(value.into())
            }
        }
    )*};
}

from_int!(u8 => U32, u16 => U32, u32 => U32, u64 => U64, i8 => I64, i16 => I64, i32 => I64, i64 => I64);

impl From<usize> for Value {
    fn from(value: usize) -> Self {
        Self::U64(value as u64)
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Vec<String>> for Value {
    fn from(value: Vec<String>) -> Self {
        Self::TextList(value)
    }
}

impl From<&[&str]> for Value {
    fn from(value: &[&str]) -> Self {
        Self::TextList(value.iter().map(|s| s.to_string()).collect())
    }
}

impl<const N: usize> From<[&str; N]> for Value {
    fn from(value: [&str; N]) -> Self {
        value.as_slice().into()
    }
}

impl From<Vec<u8>> for Value {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<&[u8]> for Value {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

/// A vector and its field values, to add with [`Db::add`](crate::Db::add).
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    vector: Vec<f32>,
    values: Vec<(String, Value)>,
}

impl Record {
    pub fn new(vector: impl Into<Vec<f32>>) -> Self {
        Self {
            vector: vector.into(),
            values: Vec::new(),
        }
    }

    /// Set the value of one field. Fields that are not set are missing.
    #[must_use]
    pub fn set(mut self, field: &str, value: impl Into<Value>) -> Self {
        self.values.push((field.to_string(), value.into()));
        self
    }

    /// Check this record against `schema` and order its values by field.
    pub(crate) fn validate(self, schema: &Schema, max_bytes: usize) -> Result<Row> {
        crate::query::validate_vector(schema.dims(), &self.vector, "vector")?;
        let mut values = vec![None; schema.len()];
        let mut size = self.vector.len() * 4;
        for (name, value) in self.values {
            let (index, ty) = schema.lookup(&name)?;
            if values[index].is_some() {
                return Err(RingDbError::InvalidRecord(format!(
                    "field '{name}' is set twice"
                )));
            }
            let value = value.coerce(ty).ok_or_else(|| {
                RingDbError::InvalidRecord(format!(
                    "field '{name}' cannot hold this value as {ty:?}"
                ))
            })?;
            size += value.stored_len();
            values[index] = Some(value);
        }
        if size > max_bytes {
            return Err(RingDbError::InvalidRecord(format!(
                "record has {size} bytes, more than the limit of {max_bytes}"
            )));
        }
        Ok(Row {
            vector: self.vector,
            values,
        })
    }
}

/// A validated record: values in schema order, coerced to their field types.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Row {
    pub(crate) vector: Vec<f32>,
    pub(crate) values: Vec<Option<Value>>,
}

/// One stored record, returned by [`Db::get`](crate::Db::get).
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub(crate) key: Key,
    pub(crate) schema: Arc<Schema>,
    pub(crate) row: Row,
}

impl Document {
    pub fn key(&self) -> Key {
        self.key
    }

    pub fn vector(&self) -> &[f32] {
        &self.row.vector
    }

    /// The value of `field`, or `None` if it is missing or not in the schema.
    pub fn get(&self, field: &str) -> Option<&Value> {
        self.row.values[self.schema.index(field)?].as_ref()
    }

    /// Fields that have a value, in schema order.
    pub fn fields(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.row
            .values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| Some((self.schema.name(index), value.as_ref()?)))
    }
}
