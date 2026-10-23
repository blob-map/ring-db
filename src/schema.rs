//! The fixed set of typed fields stored with each vector.
use crate::error::{Result, RingDbError};

/// The type of one record field.
///
/// Every field is optional on a record. Numeric, boolean, and keyword fields are
/// indexed for [`Filter`](crate::Filter)s; text, text-list, and byte fields are
/// stored only, for retrieval with [`Db::get`](crate::Db::get).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldType {
    U32,
    U64,
    I64,
    /// A finite `f32`.
    F32,
    Bool,
    /// One exact-match string, such as a city or a category.
    Keyword,
    /// Zero or more exact-match strings, such as amenities.
    Keywords,
    /// Stored UTF-8 text.
    Text,
    /// Stored list of UTF-8 strings.
    TextList,
    /// Stored bytes.
    Bytes,
}

impl FieldType {
    /// Whether filters can use this field.
    pub const fn is_filterable(self) -> bool {
        !matches!(self, Self::Text | Self::TextList | Self::Bytes)
    }

    pub(crate) const fn is_numeric(self) -> bool {
        matches!(self, Self::U32 | Self::U64 | Self::I64 | Self::F32)
    }
}

/// The vector dimension count and named fields of a database.
///
/// The schema is fixed when the database is created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    dims: usize,
    fields: Vec<(Box<str>, FieldType)>,
}

impl Schema {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            fields: Vec::new(),
        }
    }

    /// Add a field. Names must be unique and non-empty.
    #[must_use]
    pub fn field(mut self, name: &str, ty: FieldType) -> Self {
        self.fields.push((name.into(), ty));
        self
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Field names and types, in declaration order.
    pub fn fields(&self) -> impl ExactSizeIterator<Item = (&str, FieldType)> {
        self.fields.iter().map(|(name, ty)| (&**name, *ty))
    }

    pub(crate) fn len(&self) -> usize {
        self.fields.len()
    }

    pub(crate) fn ty(&self, index: usize) -> FieldType {
        self.fields[index].1
    }

    pub(crate) fn name(&self, index: usize) -> &str {
        &self.fields[index].0
    }

    pub(crate) fn index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|(field, _)| &**field == name)
    }

    /// The index and type of `name`, or an error naming the unknown field.
    pub(crate) fn lookup(&self, name: &str) -> Result<(usize, FieldType)> {
        self.index(name)
            .map(|index| (index, self.ty(index)))
            .ok_or_else(|| RingDbError::UnknownField(name.to_string()))
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.dims == 0 {
            return Err(RingDbError::InvalidSchema(
                "dimensions must be greater than zero".to_string(),
            ));
        }
        for (index, (name, _)) in self.fields.iter().enumerate() {
            if name.is_empty() {
                return Err(RingDbError::InvalidSchema(
                    "field names must not be empty".to_string(),
                ));
            }
            if self.fields[..index].iter().any(|(other, _)| other == name) {
                return Err(RingDbError::InvalidSchema(format!(
                    "field '{name}' is declared twice"
                )));
            }
        }
        Ok(())
    }
}
