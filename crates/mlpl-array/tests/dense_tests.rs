use mlpl_array::{ArrayError, DenseArray, Shape};

// -- Construction --

#[test]
fn new_vector() {
    let arr = DenseArray::new(Shape::vector(3), vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0]);
    assert_eq!(arr.rank(), 1);
}

#[test]
fn new_matrix() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(arr.rank(), 2);
    assert_eq!(arr.elem_count(), 6);
}

#[test]
fn new_data_length_mismatch() {
    let result = DenseArray::new(Shape::vector(3), vec![1.0, 2.0]);
    assert_eq!(
        result,
        Err(ArrayError::DataLengthMismatch {
            expected: 3,
            got: 2
        })
    );
}

#[test]
fn zeros_vector() {
    let arr = DenseArray::zeros(Shape::vector(4));
    assert_eq!(arr.data(), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn zeros_scalar() {
    let arr = DenseArray::zeros(Shape::scalar());
    assert_eq!(arr.data(), &[0.0]);
    assert_eq!(arr.rank(), 0);
}

#[test]
fn from_scalar() {
    let arr = DenseArray::from_scalar(42.0);
    assert_eq!(arr.shape(), &Shape::scalar());
    assert_eq!(arr.data(), &[42.0]);
}

#[test]
fn from_vec() {
    let arr = DenseArray::from_vec(vec![10.0, 20.0, 30.0]);
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[10.0, 20.0, 30.0]);
}

// -- Multi-dim indexing --

#[test]
fn get_vector() {
    let arr = DenseArray::from_vec(vec![10.0, 20.0, 30.0]);
    assert_eq!(arr.get(&[0]).unwrap(), &10.0);
    assert_eq!(arr.get(&[2]).unwrap(), &30.0);
}

#[test]
fn get_matrix_row_major() {
    // shape [2, 3], data [1,2,3,4,5,6]
    // row 0: [1, 2, 3], row 1: [4, 5, 6]
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(arr.get(&[0, 0]).unwrap(), &1.0);
    assert_eq!(arr.get(&[0, 2]).unwrap(), &3.0);
    assert_eq!(arr.get(&[1, 0]).unwrap(), &4.0);
    assert_eq!(arr.get(&[1, 2]).unwrap(), &6.0);
}

#[test]
fn get_scalar() {
    let arr = DenseArray::from_scalar(5.0);
    assert_eq!(arr.get(&[]).unwrap(), &5.0);
}

#[test]
fn get_out_of_bounds() {
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(
        arr.get(&[3]),
        Err(ArrayError::IndexOutOfBounds {
            axis: 0,
            index: 3,
            size: 3
        })
    );
}

#[test]
fn get_rank_mismatch() {
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(
        arr.get(&[0, 0]),
        Err(ArrayError::RankMismatch {
            expected: 1,
            got: 2
        })
    );
}

#[test]
fn get_empty_array() {
    let arr = DenseArray::zeros(Shape::vector(0));
    assert_eq!(arr.get(&[0]), Err(ArrayError::EmptyArray));
}

// -- Set --

#[test]
fn set_vector() {
    let mut arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    arr.set(&[1], 99.0).unwrap();
    assert_eq!(arr.get(&[1]).unwrap(), &99.0);
}

#[test]
fn set_matrix() {
    let mut arr = DenseArray::new(Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    arr.set(&[1, 0], 99.0).unwrap();
    assert_eq!(arr.get(&[1, 0]).unwrap(), &99.0);
}

#[test]
fn set_out_of_bounds() {
    let mut arr = DenseArray::from_vec(vec![1.0, 2.0]);
    assert_eq!(
        arr.set(&[5], 0.0),
        Err(ArrayError::IndexOutOfBounds {
            axis: 0,
            index: 5,
            size: 2
        })
    );
}

// -- Display --

#[test]
fn display_scalar() {
    let arr = DenseArray::from_scalar(7.5);
    assert_eq!(arr.to_string(), "7.5");
}

#[test]
fn display_vector() {
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(arr.to_string(), "1 2 3");
}

#[test]
fn display_matrix() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(arr.to_string(), "1 2 3\n4 5 6");
}

#[test]
fn display_empty_vector() {
    let arr = DenseArray::zeros(Shape::vector(0));
    assert_eq!(arr.to_string(), "[]");
}
