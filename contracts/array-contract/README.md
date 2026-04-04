# Array Contract

## Purpose

Define the first contract for dense arrays and tensors in MLPL.

This contract is intentionally small. It exists to support low-context implementation tasks in `mlpl-array`.

## Concepts

### Shape
A shape is an ordered list of non-negative dimensions.

### Rank
Rank is the number of dimensions in the shape.

### Element count
The total element count is the product of the shape dimensions.

## Invariants

- shape order matters
- rank equals `shape.len()`
- element count must be deterministic
- operations must report explicit errors on invalid input
- reshape preserves element order
- reshape succeeds only when source and target element counts match

## Initial operations

- shape creation
- rank query
- dimension access
- element count query
- reshape
- transpose

## Errors

The first implementation should favor explicit error variants over generic string errors.

## Open questions

- whether zero dimensions are allowed in all contexts
- whether shapes should use `usize` or a narrower type
- whether array views and strides are first-class in v1
