use std::ops::{Add, Div, Index, Mul, Sub};
use std::simd::Simd;
use std::simd::num::SimdFloat;

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Vector<const N: usize> {
    inner: Simd<f32, N>,
}
macro_rules! impl_elementwise_ops {
    ($($trait:ident => $method:ident),*) => {
        $(
            impl<const N: usize> $trait for Vector<N> {
                type Output = Vector<N>;
                fn $method(self, rhs: Self) -> Self::Output {
                    Self { inner: self.inner.$method(rhs.inner) }
                }
            }
            impl<const N: usize> $trait<&Self> for Vector<N> {
                type Output = Vector<N>;
                fn $method(self, rhs: &Self) -> Self::Output {
                    Self { inner: self.inner.$method(rhs.inner) }
                }
            }
             impl<const N: usize> $trait for &Vector<N> {
                type Output = Vector<N>;
                fn $method(self, rhs: Self) -> Self::Output {
                    Vector { inner: self.inner.$method(rhs.inner) }
                }
            }
            impl<const N: usize> $trait<&Self> for &Vector<N> {
                type Output = Vector<N>;
                fn $method(self, rhs: &Self) -> Self::Output {
                    Vector { inner: self.inner.$method(rhs.inner) }
                }
            }
        )*
    };
}
impl_elementwise_ops!(Add => add, Sub => sub, Mul => mul, Div => div);
impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        let val = &self.inner[index];
        &val
    }
}
impl<const N: usize> Vector<N> {
    pub fn new(data: [f32; N]) -> Self {
        let inner = Simd::from_slice(data.as_slice());
        Self { inner }
    }
    pub fn dot(&self, other: &Self) -> f32 {
        (self * other).inner.reduce_sum()
    }
    pub fn prod(&self) -> f32 {
        self.inner.reduce_product()
    }
    pub fn sum(&self) -> f32 {
        self.inner.reduce_sum()
    }
}
impl Vector<3> {
    pub fn cross(&self, other: &Self) -> Self {
        let a = self.inner;
        let b = other.inner;
        let a_yzx = Simd::from_array([a[1], a[2], a[0]]);
        let a_zxy = Simd::from_array([a[2], a[0], a[1]]);
        let b_yzx = Simd::from_array([b[1], b[2], b[0]]);
        let b_zxy = Simd::from_array([b[2], b[0], b[1]]);
        let result = a_yzx * b_zxy - a_zxy * b_yzx;
        Self { inner: result }
    }
}
