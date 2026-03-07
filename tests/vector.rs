use quadrax::cpu::maths::vector::Vector;

#[pollster::test]
async fn add_prod_dot_cross() {
    let v1 = Vector::new([1.0, 2.0, 3.0]);
    let v2 = Vector::new([1.0; 3]);
    assert_eq!(v1 + v2, Vector::new([2.0, 3.0, 4.0]));
    assert_eq!(v1.dot(&v2), (v1 * v2).sum());
    assert_eq!(v1.cross(&v2), Vector::new([-1.0, 2.0, -1.0]));
}
