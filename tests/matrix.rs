use quadrax::cpu::maths::matrix::Matrix;

#[pollster::test]
async fn matrix_operations_various_shapes() {
    let row1 = Matrix::<3, 1>::new([[1.0, 2.0, 3.0]]);
    let row2 = Matrix::<3, 1>::new([[1.0, 1.0, 1.0]]);

    assert_eq!(row1 + row2, Matrix::<3, 1>::new([[2.0, 3.0, 4.0]]));
    assert_eq!((row1 * row2).sum(), row1.dot(&row2));

    let col1 = Matrix::<1, 3>::new([[1.0], [2.0], [3.0]]);
    let col2 = Matrix::<1, 3>::new([[1.0], [1.0], [1.0]]);

    assert_eq!(col1 + col2, Matrix::<1, 3>::new([[2.0], [3.0], [4.0]]));
    assert_eq!(col1.dot(&col2), (col1 * col2).sum());

    let a = Matrix::<3, 2>::new([[1., 2., 3.], [4., 5., 6.]]);
    let b = Matrix::<2, 3>::new([[7., 8.], [9., 10.], [11., 12.]]);

    let c = a | b;
    let expected = Matrix::<2, 2>::new([[58., 64.], [139., 154.]]);
    assert_eq!(c, expected);

    let sum = a + a;
    let prod = a * a;
    let total_sum = sum.sum();
    let total_prod = prod.prod();
    assert!(total_sum > 0.0);
    assert!(total_prod > 0.0);

    let dot = a.dot(&a);
    assert!(dot > 0.0);
}
