use mlpl_array::{ArrayError, DenseArray, Shape};

// -- Reshape --

#[test]
fn reshape_vector_to_matrix() {
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let reshaped = arr.reshape(Shape::new(vec![2, 3])).unwrap();
    assert_eq!(reshaped.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(reshaped.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn reshape_matrix_to_vector() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let reshaped = arr.reshape(Shape::vector(6)).unwrap();
    assert_eq!(reshaped.shape(), &Shape::vector(6));
    assert_eq!(reshaped.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn reshape_matrix_to_different_matrix() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let reshaped = arr.reshape(Shape::new(vec![3, 2])).unwrap();
    assert_eq!(reshaped.shape(), &Shape::new(vec![3, 2]));
    // Element order preserved (row-major reinterpretation)
    assert_eq!(reshaped.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn reshape_preserves_element_order() {
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let reshaped = arr.reshape(Shape::new(vec![2, 2])).unwrap();
    // [1,2,3,4] as [2,2] -> row 0: [1,2], row 1: [3,4]
    assert_eq!(reshaped.get(&[0, 0]).unwrap(), &1.0);
    assert_eq!(reshaped.get(&[0, 1]).unwrap(), &2.0);
    assert_eq!(reshaped.get(&[1, 0]).unwrap(), &3.0);
    assert_eq!(reshaped.get(&[1, 1]).unwrap(), &4.0);
}

#[test]
fn reshape_incompatible_error() {
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    let result = arr.reshape(Shape::new(vec![2, 2]));
    assert_eq!(
        result,
        Err(ArrayError::ShapeMismatch {
            source: 3,
            target: 4
        })
    );
}

#[test]
fn reshape_scalar_to_vector() {
    let arr = DenseArray::from_scalar(42.0);
    let reshaped = arr.reshape(Shape::vector(1)).unwrap();
    assert_eq!(reshaped.shape(), &Shape::vector(1));
    assert_eq!(reshaped.data(), &[42.0]);
}

// -- Transpose --

#[test]
fn transpose_matrix() {
    // shape [2, 3], data [1,2,3,4,5,6]
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t = arr.transpose();
    assert_eq!(t.shape(), &Shape::new(vec![3, 2]));
    assert_eq!(t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn transpose_square_matrix() {
    // [[1,2],[3,4]] -> [[1,3],[2,4]]
    let arr = DenseArray::new(Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let t = arr.transpose();
    assert_eq!(t.shape(), &Shape::new(vec![2, 2]));
    assert_eq!(t.data(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn transpose_vector_identity() {
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    let t = arr.transpose();
    assert_eq!(t.shape(), &Shape::vector(3));
    assert_eq!(t.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn transpose_scalar_identity() {
    let arr = DenseArray::from_scalar(7.0);
    let t = arr.transpose();
    assert_eq!(t.shape(), &Shape::scalar());
    assert_eq!(t.data(), &[7.0]);
}

#[test]
fn transpose_3d() {
    // shape [2, 3, 1], reversed -> [1, 3, 2]
    let arr = DenseArray::new(
        Shape::new(vec![2, 3, 1]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let t = arr.transpose();
    assert_eq!(t.shape(), &Shape::new(vec![1, 3, 2]));
    // Verify element at original [i,j,k] is at transposed [k,j,i]
    // orig [0,0,0]=1, [0,1,0]=2, [0,2,0]=3, [1,0,0]=4, [1,1,0]=5, [1,2,0]=6
    // trans [0,0,0]=1, [0,0,1]=4, [0,1,0]=2, [0,1,1]=5, [0,2,0]=3, [0,2,1]=6
    assert_eq!(t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

// -- Round-trip --

#[test]
fn reshape_then_transpose() {
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let matrix = arr.reshape(Shape::new(vec![2, 3])).unwrap();
    let transposed = matrix.transpose();
    assert_eq!(transposed.shape(), &Shape::new(vec![3, 2]));
    // All original values present
    let mut sorted = transposed.data().to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn double_transpose_identity() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t2 = arr.transpose().transpose();
    assert_eq!(t2.shape(), arr.shape());
    assert_eq!(t2.data(), arr.data());
}
