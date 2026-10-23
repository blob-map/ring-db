//! Typed filters over record fields.
use std::ops::{BitAnd, BitOr, Bound, Not, RangeBounds, RangeInclusive};

use crate::column::Column;
use crate::error::{Result, RingDbError};
use crate::query::Key;
use crate::rows::RowSet;
use crate::schema::{FieldType, Schema};
use crate::segment::Segment;
use crate::value::Value;

/// A condition on record fields that restricts a search.
///
/// Combine filters with `&`, `|`, and `!`. A record with no value for a field
/// does not match a condition on that field, so `!Filter::eq(f, v)` includes
/// records where `f` is missing.
///
/// ```
/// use ringdb::Filter;
///
/// let filter = Filter::eq("city", "Lyon")
///     & Filter::range("price", 100_000..=400_000)
///     & !Filter::any_of("tags", ["ground-floor", "no-elevator"]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Filter(Node);

#[derive(Debug, Clone, PartialEq)]
enum Node {
    Eq(String, Value),
    AnyOf(String, Vec<Value>),
    Range(String, Bound<Value>, Bound<Value>),
    Keys(Vec<Key>),
    And(Vec<Node>),
    Or(Vec<Node>),
    Not(Box<Node>),
}

impl Filter {
    /// Records whose `field` equals `value`. On a `Keywords` field, records
    /// that contain `value`.
    pub fn eq(field: &str, value: impl Into<Value>) -> Self {
        Self(Node::Eq(field.to_string(), value.into()))
    }

    /// Records whose `field` equals, or contains, at least one of `values`.
    pub fn any_of<V: Into<Value>>(field: &str, values: impl IntoIterator<Item = V>) -> Self {
        Self(Node::AnyOf(
            field.to_string(),
            values.into_iter().map(Into::into).collect(),
        ))
    }

    /// Records whose numeric `field` is in `range`.
    pub fn range<V: Into<Value> + Clone>(field: &str, range: impl RangeBounds<V>) -> Self {
        let bound = |bound: Bound<&V>| bound.map(|value| value.clone().into());
        Self(Node::Range(
            field.to_string(),
            bound(range.start_bound()),
            bound(range.end_bound()),
        ))
    }

    /// Records with one of `keys`.
    pub fn keys(keys: impl IntoIterator<Item = Key>) -> Self {
        let mut keys: Vec<Key> = keys.into_iter().collect();
        keys.sort_unstable();
        keys.dedup();
        Self(Node::Keys(keys))
    }

    /// Check field names and value types against `schema`.
    pub(crate) fn compile(&self, schema: &Schema) -> Result<Compiled> {
        self.0.compile(schema)
    }
}

impl BitAnd for Filter {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        Self(match (self.0, other.0) {
            (Node::And(mut left), Node::And(right)) => {
                left.extend(right);
                Node::And(left)
            }
            (Node::And(mut nodes), node) | (node, Node::And(mut nodes)) => {
                nodes.push(node);
                Node::And(nodes)
            }
            (left, right) => Node::And(vec![left, right]),
        })
    }
}

impl BitOr for Filter {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(match (self.0, other.0) {
            (Node::Or(mut left), Node::Or(right)) => {
                left.extend(right);
                Node::Or(left)
            }
            (Node::Or(mut nodes), node) | (node, Node::Or(mut nodes)) => {
                nodes.push(node);
                Node::Or(nodes)
            }
            (left, right) => Node::Or(vec![left, right]),
        })
    }
}

impl Not for Filter {
    type Output = Self;

    fn not(self) -> Self {
        Self(match self.0 {
            Node::Not(inner) => *inner,
            node => Node::Not(Box::new(node)),
        })
    }
}

/// A filter checked against a schema, ready to evaluate on segments.
#[derive(Debug)]
pub(crate) enum Compiled {
    Int {
        field: usize,
        low: i128,
        high: i128,
    },
    Float {
        field: usize,
        low: Bound<f32>,
        high: Bound<f32>,
    },
    Bool {
        field: usize,
        value: bool,
    },
    Terms {
        field: usize,
        terms: Vec<String>,
    },
    Keys(Vec<Key>),
    And(Vec<Compiled>),
    Or(Vec<Compiled>),
    Not(Box<Compiled>),
}

fn invalid(field: &str, message: &str) -> RingDbError {
    RingDbError::InvalidQuery(format!("filter on '{field}': {message}"))
}

impl Node {
    fn compile(&self, schema: &Schema) -> Result<Compiled> {
        Ok(match self {
            Self::Eq(name, value) => equal(schema, name, value)?,
            Self::AnyOf(name, values) => {
                let (field, ty) = filterable(schema, name)?;
                if matches!(ty, FieldType::Keyword | FieldType::Keywords) {
                    let terms = values
                        .iter()
                        .map(|value| term(name, value))
                        .collect::<Result<_>>()?;
                    Compiled::Terms { field, terms }
                } else {
                    let nodes = values
                        .iter()
                        .map(|value| equal(schema, name, value))
                        .collect::<Result<_>>()?;
                    Compiled::Or(nodes)
                }
            }
            Self::Range(name, low, high) => range(schema, name, low, high)?,
            Self::Keys(keys) => Compiled::Keys(keys.clone()),
            Self::And(nodes) => Compiled::And(compile_all(nodes, schema)?),
            Self::Or(nodes) => Compiled::Or(compile_all(nodes, schema)?),
            Self::Not(node) => Compiled::Not(Box::new(node.compile(schema)?)),
        })
    }
}

fn compile_all(nodes: &[Node], schema: &Schema) -> Result<Vec<Compiled>> {
    nodes.iter().map(|node| node.compile(schema)).collect()
}

fn filterable(schema: &Schema, name: &str) -> Result<(usize, FieldType)> {
    let (field, ty) = schema.lookup(name)?;
    if !ty.is_filterable() {
        return Err(invalid(name, &format!("{ty:?} fields are stored only")));
    }
    Ok((field, ty))
}

fn term(name: &str, value: &Value) -> Result<String> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| invalid(name, "keyword values must be strings"))
}

fn equal(schema: &Schema, name: &str, value: &Value) -> Result<Compiled> {
    let (field, ty) = filterable(schema, name)?;
    if ty.is_numeric() {
        let bound = Bound::Included(value.clone());
        return range(schema, name, &bound, &bound);
    }
    Ok(match ty {
        FieldType::Bool => Compiled::Bool {
            field,
            value: value
                .as_bool()
                .ok_or_else(|| invalid(name, "the value must be a bool"))?,
        },
        _ => Compiled::Terms {
            field,
            terms: vec![term(name, value)?],
        },
    })
}

fn range(schema: &Schema, name: &str, low: &Bound<Value>, high: &Bound<Value>) -> Result<Compiled> {
    let (field, ty) = filterable(schema, name)?;
    if !ty.is_numeric() {
        return Err(invalid(name, &format!("{ty:?} fields have no order")));
    }
    if ty == FieldType::F32 {
        let float = |bound: &Bound<Value>| -> Result<Bound<f32>> {
            let convert = |value: &Value| {
                value
                    .as_f32()
                    .or_else(|| value.as_int().map(|v| v as f32))
                    .filter(|v| !v.is_nan())
                    .ok_or_else(|| invalid(name, "bounds must be numbers"))
            };
            Ok(match bound {
                Bound::Included(v) => Bound::Included(convert(v)?),
                Bound::Excluded(v) => Bound::Excluded(convert(v)?),
                Bound::Unbounded => Bound::Unbounded,
            })
        };
        return Ok(Compiled::Float {
            field,
            low: float(low)?,
            high: float(high)?,
        });
    }
    let int = |bound: &Bound<Value>, unbounded: i128, step: i128| -> Result<i128> {
        let convert = |value: &Value| {
            value
                .as_int()
                .ok_or_else(|| invalid(name, "bounds of an integer field must be integers"))
        };
        Ok(match bound {
            Bound::Included(v) => convert(v)?,
            Bound::Excluded(v) => convert(v)? + step,
            Bound::Unbounded => unbounded,
        })
    };
    Ok(Compiled::Int {
        field,
        low: int(low, i128::MIN, 1)?,
        high: int(high, i128::MAX, -1)?,
    })
}

impl Compiled {
    /// The rows of `segment` that match.
    pub(crate) fn rows(&self, segment: &Segment) -> RowSet {
        let n = segment.len();
        match self {
            Self::Int { field, low, high } => match &segment.columns[*field] {
                Column::U32(c) => int_range(*low, *high, u32::MIN, u32::MAX)
                    .map_or_else(|| RowSet::empty(n), |r| c.select(|v| r.contains(&v))),
                Column::U64(c) => int_range(*low, *high, u64::MIN, u64::MAX)
                    .map_or_else(|| RowSet::empty(n), |r| c.select(|v| r.contains(&v))),
                Column::I64(c) => int_range(*low, *high, i64::MIN, i64::MAX)
                    .map_or_else(|| RowSet::empty(n), |r| c.select(|v| r.contains(&v))),
                _ => unreachable!("integer filters apply to integer columns"),
            },
            Self::Float { field, low, high } => match &segment.columns[*field] {
                Column::F32(c) => c.select(|v| (*low, *high).contains(&v)),
                _ => unreachable!("float filters apply to float columns"),
            },
            Self::Bool { field, value } => match &segment.columns[*field] {
                Column::Bool(c) => c.select(|v| v == *value),
                _ => unreachable!("bool filters apply to bool columns"),
            },
            Self::Terms { field, terms } => match &segment.columns[*field] {
                Column::Keywords(c) => c.select(terms),
                _ => unreachable!("term filters apply to keyword columns"),
            },
            Self::Keys(keys) => {
                let (first, last) = (segment.keys[0], segment.keys[n - 1]);
                let start = keys.partition_point(|&key| key < first);
                let end = keys.partition_point(|&key| key <= last);
                let rows = keys[start..end]
                    .iter()
                    .filter_map(|&key| segment.row(key))
                    .collect();
                RowSet::from_sorted(n, rows)
            }
            Self::And(nodes) => {
                let mut rows = RowSet::all(n);
                for node in nodes {
                    if rows.len() == 0 {
                        break;
                    }
                    rows = rows.intersection(node.rows(segment));
                }
                rows
            }
            Self::Or(nodes) => nodes
                .iter()
                .map(|node| node.rows(segment))
                .fold(RowSet::empty(n), RowSet::union),
            Self::Not(node) => node.rows(segment).complement(),
        }
    }
}

/// The inclusive `low..=high` range clamped to `T`, or `None` if it is empty.
fn int_range<T>(low: i128, high: i128, min: T, max: T) -> Option<RangeInclusive<T>>
where
    T: Copy + Into<i128> + TryFrom<i128>,
{
    let low = T::try_from(low.max(min.into())).ok()?;
    let high = T::try_from(high.min(max.into())).ok()?;
    (low.into() <= high.into()).then_some(low..=high)
}
