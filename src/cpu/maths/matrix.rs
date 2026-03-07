use std::ops::BitOr;

use crate::cpu::maths::vector::Vector;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Matrix<const NX: usize, const NY: usize> {
    pub inner: [Vector<NX>; NY],
}

macro_rules! impl_elementwise_ops {
    ($($trait:ident => $method:ident),*) => {
        $(
            impl<const NX: usize, const NY: usize> std::ops::$trait for Matrix<NX, NY> {
                type Output = Self;

                fn $method(self, rhs: Self) -> Self::Output {
                    let inner = std::array::from_fn(|i| self.inner[i].$method(rhs.inner[i]));
                    Self { inner }
                }
            }

            impl<const NX: usize, const NY: usize> std::ops::$trait<&Self> for Matrix<NX, NY> {
                type Output = Self;

                fn $method(self, rhs: &Self) -> Self::Output {
                    let inner = std::array::from_fn(|i| self.inner[i].$method(rhs.inner[i]));
                    Self { inner }
                }
            }

            impl<const NX: usize, const NY: usize> std::ops::$trait for &Matrix<NX, NY> {
                type Output = Matrix<NX, NY>;

                fn $method(self, rhs: Self) -> Self::Output {
                    let inner = std::array::from_fn(|i| self.inner[i].$method(rhs.inner[i]));
                    Matrix { inner }
                }
            }

            impl<const NX: usize, const NY: usize> std::ops::$trait<&Self> for &Matrix<NX, NY> {
                type Output = Matrix<NX, NY>;

                fn $method(self, rhs: &Self) -> Self::Output {
                    let inner = std::array::from_fn(|i| self.inner[i].$method(rhs.inner[i]));
                    Matrix { inner }
                }
            }
        )*
    };
}

impl_elementwise_ops!(Add => add, Sub => sub, Mul => mul, Div => div);

impl<const NX: usize, const NY: usize> Matrix<NX, NY> {
    pub fn new(data: [[f32; NX]; NY]) -> Self {
        let inner = data.map(Vector::new);
        Self { inner }
    }
    pub fn sum(&self) -> f32 {
        self.inner.iter().map(|v| v.sum()).sum()
    }
    pub fn prod(&self) -> f32 {
        self.inner.iter().map(|v| v.prod()).product()
    }
    pub fn dot(&self, other: &Self) -> f32 {
        self.inner
            .iter()
            .zip(other.inner.iter())
            .map(|(a, b)| a.dot(b))
            .sum()
    }
    pub fn transpose(&self) -> Matrix<NY, NX> {
        let mut data = [[0.0_f32; NY]; NX];
        for i in 0..NY {
            for j in 0..NX {
                data[j][i] = self.inner[i][j]; // swap rows and columns
            }
        }
        Matrix::new(data)
    }
    pub fn row(&self, i: usize) -> &Vector<NX> {
        &self.inner[i]
    }
}

impl<const NX: usize, const NY: usize, const NZ: usize> BitOr<Matrix<NZ, NX>> for Matrix<NX, NY> {
    type Output = Matrix<NZ, NY>;
    fn bitor(self, rhs: Matrix<NZ, NX>) -> Self::Output {
        let rhs_t = rhs.transpose();
        let inner =
            std::array::from_fn(|i| std::array::from_fn(|j| self.row(i).dot(&rhs_t.row(j))));
        Matrix {
            inner: inner.map(Vector::new),
        }
    }
}
