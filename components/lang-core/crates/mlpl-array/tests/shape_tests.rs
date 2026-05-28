use mlpl_array::Shape;

#[test]
fn scalar_shape() {
    let s = Shape::scalar();
    assert_eq!(s.rank(), 0);
    assert_eq!(s.dims(), &[]);
    assert_eq!(s.elem_count(), 1);
}

#[test]
fn vector_shape() {
    let s = Shape::vector(5);
    assert_eq!(s.rank(), 1);
    assert_eq!(s.dims(), &[5]);
    assert_eq!(s.elem_count(), 5);
}

#[test]
fn matrix_shape() {
    let s = Shape::new(vec![2, 3]);
    assert_eq!(s.rank(), 2);
    assert_eq!(s.dims(), &[2, 3]);
    assert_eq!(s.elem_count(), 6);
}

#[test]
fn new_3d() {
    let s = Shape::new(vec![2, 3, 4]);
    assert_eq!(s.rank(), 3);
    assert_eq!(s.dims(), &[2, 3, 4]);
    assert_eq!(s.elem_count(), 24);
}

#[test]
fn zero_dim_shape() {
    let s = Shape::new(vec![2, 0, 3]);
    assert_eq!(s.rank(), 3);
    assert_eq!(s.elem_count(), 0);
}

#[test]
fn single_element_matrix() {
    let s = Shape::new(vec![1, 1]);
    assert_eq!(s.elem_count(), 1);
    assert_eq!(s.rank(), 2);
}

#[test]
fn display_scalar() {
    assert_eq!(Shape::scalar().to_string(), "[]");
}

#[test]
fn display_vector() {
    assert_eq!(Shape::vector(5).to_string(), "[5]");
}

#[test]
fn display_matrix() {
    assert_eq!(Shape::new(vec![2, 3]).to_string(), "[2, 3]");
}

#[test]
fn display_3d() {
    assert_eq!(Shape::new(vec![2, 3, 4]).to_string(), "[2, 3, 4]");
}

#[test]
fn equality() {
    assert_eq!(Shape::new(vec![2, 3]), Shape::new(vec![2, 3]));
    assert_ne!(Shape::new(vec![2, 3]), Shape::new(vec![3, 2]));
}

#[test]
fn clone_shape() {
    let a = Shape::new(vec![2, 3]);
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn vector_zero_length() {
    let s = Shape::vector(0);
    assert_eq!(s.rank(), 1);
    assert_eq!(s.elem_count(), 0);
}
