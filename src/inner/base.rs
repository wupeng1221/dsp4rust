use ndarray::iter::{Iter, IterMut};
use ndarray::{concatenate, Axis};
use ndarray::{Array1, Ix, Ix1};
use ndarray_stats::errors::MinMaxError;
use ndarray_stats::QuantileExt;
use num_traits::AsPrimitive;
use std::fmt::{Display, Formatter};
use std::iter::repeat;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use super::{BaseOperationError, PadError};

#[derive(Debug, Clone)]
pub struct SignalBase {
    base: Array1<f64>,
}

// 实现 Display
impl Display for SignalBase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.base.to_vec())
    }
}
// 从迭代器收集
impl<A> FromIterator<A> for SignalBase
where
    A: AsPrimitive<f64>,
{
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        SignalBase {
            base: Array1::from_iter(iter.into_iter().map(AsPrimitive::as_)),
        }
    }
}
// 实现 Iterator, 也会自动实现 IntoIterator
impl Iterator for SignalBase {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.base.iter().next().copied()
    }
}
// 实现滑动窗口
impl SignalBase {
    pub fn windows(&self, window_size: usize) -> impl Iterator<Item = SignalBase> + '_ {
        self.base
            .windows(window_size)
            .into_iter()
            .map(|window| SignalBase::from_array1(window.to_owned()))
    }
}
// 实现方括号索引
impl Index<isize> for SignalBase {
    type Output = f64;
    fn index(&self, index: isize) -> &Self::Output {
        if !self.is_idx_valid(index) {
            panic!("Index out of bound");
        }
        let idx = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &self.base[idx]
    }
}
// 可变索引操作
impl IndexMut<isize> for SignalBase {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        if !self.is_idx_valid(index) {
            panic!("Index out of bound");
        }
        let idx = if index < 0 {
            (self.len() as isize + index) as usize
        } else {
            index as usize
        };
        &mut self.base[idx]
    }
}
// 构造函数
impl SignalBase {
    fn from_array1(base: Array1<f64>) -> Self {
        SignalBase { base }
    }
    pub fn from_vec(signal: Vec<f64>) -> Self {
        Self::from_array1(Array1::from_vec(signal))
    }

    pub fn from_elem(ele: f64, len: usize) -> Self {
        Self::from_array1(Array1::from_elem(len, ele))
    }

    pub fn from_len_fn<F>(len: usize, f: F) -> Self
    where
        F: Fn(usize) -> f64,
    {
        Self::from_array1(Array1::from_shape_fn(len, f))
    }

    pub fn from_iter<I, T>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: AsPrimitive<f64>,
    {
        Self::from_array1(Array1::from_iter(iter.into_iter().map(AsPrimitive::as_)))
    }

    pub fn zeros(len: usize) -> Self {
        Self::from_array1(Array1::zeros(len))
    }

    pub fn ones(len: usize) -> Self {
        Self::from_array1(Array1::ones(len))
    }

    pub fn linspace(start: f64, end: f64, len: usize) -> Self {
        Self::from_array1(Array1::linspace(start, end, len))
    }

    pub fn arrange(start: f64, end_exclude: f64, step: f64) -> Self {
        Self::from_array1(Array1::range(start, end_exclude, step))
    }
}
// 作为对象的一些基本性质的实现
impl SignalBase {
    pub fn iter(&self) -> Iter<'_, f64, Ix1> {
        self.base.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, f64, Ix1> {
        self.base.iter_mut()
    }

    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: FnMut(&mut f64),
    {
        self.base.map_inplace(f)
    }

    pub fn map<F>(&self, f: F) -> Self
    where
        F: FnMut(&f64) -> f64,
    {
        SignalBase {
            base: self.base.map(f),
        }
    }
}
// 转换成数组切片
impl SignalBase {
    pub fn as_slice(&self) -> Option<&[f64]> {
        self.base.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> Option<&mut [f64]> {
        self.base.as_slice_mut()
    }
}
// 信号的统计信息实现
impl SignalBase {
    pub fn len(&self) -> usize {
        self.base.len()
    }

    pub fn sum(&self) -> f64 {
        self.base.iter().sum()
    }
    pub fn min(&self) -> Result<f64, MinMaxError> {
        Ok(*self.base.min()?)
    }

    pub fn argmin(&self) -> Result<Ix, MinMaxError> {
        self.base.argmin()
    }

    pub fn max(&self) -> Result<f64, MinMaxError> {
        Ok(*self.base.max()?)
    }

    pub fn argmax(&self) -> Result<Ix, MinMaxError> {
        self.base.argmax()
    }

    pub fn mean(&self) -> Option<f64> {
        self.base.mean()
    }

    pub fn std_pop(&self) -> f64 {
        self.base.std(0.)
    }

    pub fn std_sample(&self) -> f64 {
        self.base.std(1.)
    }

    pub fn var_pop(&self) -> f64 {
        self.base.var(0.)
    }

    pub fn var_sample(&self) -> f64 {
        self.base.var(1.)
    }

    pub fn range(&self) -> Result<(f64, f64), MinMaxError> {
        let min = self.min()?;
        let max = self.max()?;
        Ok((min, max))
    }

    pub fn p2p(&self) -> Result<f64, MinMaxError> {
        Ok(self.max()? - self.min()?)
    }

    pub fn energy(&self) -> f64 {
        self.base.iter().map(|&x| x.powf(2.)).sum::<f64>()
    }

    pub fn avg_power(&self) -> f64 {
        self.energy() / self.len() as f64
    }
}
// 信号截取生成新的信号
impl SignalBase {
    fn is_idx_valid(&self, idx: isize) -> bool {
        let len = self.len();
        idx >= -(len as isize) && idx < len as isize
    }
    pub fn cut_from(&self, from: isize) -> SignalBase {
        if !self.is_idx_valid(from) {
            panic!("Index out of bounds");
        }
        let n_skip = self.pos_idx(from);
        let iter = self.iter().skip(n_skip).cloned();
        SignalBase {
            base: Array1::from_iter(iter),
        }
    }

    pub fn cut_to(&self, to: isize) -> SignalBase {
        if !self.is_idx_valid(to) {
            panic!("Index out of bounds");
        }
        let n_skip = self.len() - self.pos_idx(to) - 1;
        let iter = self.iter().rev().skip(n_skip).rev().cloned();
        SignalBase {
            base: Array1::from_iter(iter),
        }
    }

    fn pos_idx(&self, index: isize) -> usize {
        if index < 0 {
            self.len() + index as usize
        } else {
            index as usize
        }
    }
    pub fn cut_from_to(&self, from: isize, to: isize) -> SignalBase {
        if !self.is_idx_valid(from) || !self.is_idx_valid(to) {
            panic!("Index out of bounds");
        }
        if self.pos_idx(from) > self.pos_idx(to) {
            panic!("Invalid range, the positive index of from must be less than to's");
        }
        let head_skip = self.pos_idx(from);
        let tail_skip = self.len() - self.pos_idx(to) - 1;
        let iter = self
            .iter()
            .skip(head_skip)
            .rev()
            .skip(tail_skip)
            .rev()
            .cloned();
        SignalBase {
            base: Array1::from_iter(iter),
        }
    }
}
// tovec
impl SignalBase {
    pub fn to_vec(&self) -> Vec<f64> {
        self.base.to_vec()
    }
}
// 实现自身的加减乘除，只保留引用传递
impl<'a, 'b> Add<&'b SignalBase> for &'a SignalBase {
    type Output = SignalBase;
    fn add(self, other: &'b SignalBase) -> SignalBase {
        SignalBase {
            base: &self.base + &other.base,
        }
    }
}

impl<'a, 'b> Sub<&'b SignalBase> for &'a SignalBase {
    type Output = SignalBase;
    fn sub(self, other: &'b SignalBase) -> SignalBase {
        SignalBase {
            base: &self.base - &other.base,
        }
    }
}

impl<'a, 'b> Mul<&'b SignalBase> for &'a SignalBase {
    type Output = SignalBase;
    fn mul(self, other: &'b SignalBase) -> SignalBase {
        SignalBase {
            base: &self.base * &other.base,
        }
    }
}

impl<'a, 'b> Div<&'b SignalBase> for &'a SignalBase {
    type Output = SignalBase;
    fn div(self, other: &'b SignalBase) -> SignalBase {
        SignalBase {
            base: &self.base / &other.base,
        }
    }
}

// 实现 += -= *= /=，只保留引用传递
impl<'a> AddAssign<&'a SignalBase> for SignalBase {
    fn add_assign(&mut self, other: &'a SignalBase) {
        self.base += &other.base;
    }
}

impl<'a> SubAssign<&'a SignalBase> for SignalBase {
    fn sub_assign(&mut self, other: &'a SignalBase) {
        self.base -= &other.base;
    }
}

impl<'a> MulAssign<&'a SignalBase> for SignalBase {
    fn mul_assign(&mut self, other: &'a SignalBase) {
        self.base *= &other.base;
    }
}

impl<'a> DivAssign<&'a SignalBase> for SignalBase {
    fn div_assign(&mut self, other: &'a SignalBase) {
        self.base /= &other.base;
    }
}

// 实现和标量的广播，只保留引用传递，标量只能出现在右侧
impl<'a, 'b, T> Add<&'b T> for &'a SignalBase
where
    T: AsPrimitive<f64>,
{
    type Output = SignalBase;
    fn add(self, other: &'b T) -> SignalBase {
        SignalBase {
            base: &self.base + other.as_(),
        }
    }
}

impl<'a, 'b, T> Sub<&'b T> for &'a SignalBase
where
    T: AsPrimitive<f64>,
{
    type Output = SignalBase;
    fn sub(self, other: &'b T) -> SignalBase {
        SignalBase {
            base: &self.base - other.as_(),
        }
    }
}

impl<'a, 'b, T> Mul<&'b T> for &'a SignalBase
where
    T: AsPrimitive<f64>,
{
    type Output = SignalBase;
    fn mul(self, other: &'b T) -> SignalBase {
        SignalBase {
            base: &self.base * other.as_(),
        }
    }
}

impl<'a, 'b, T> Div<&'b T> for &'a SignalBase
where
    T: AsPrimitive<f64>,
{
    type Output = SignalBase;
    fn div(self, other: &'b T) -> SignalBase {
        SignalBase {
            base: &self.base / other.as_(),
        }
    }
}

// 为 SignalBase 实现与引用标量的原地运算
impl<'a, T> AddAssign<&'a T> for SignalBase
where
    T: AsPrimitive<f64>,
{
    fn add_assign(&mut self, other: &'a T) {
        self.base += other.as_();
    }
}

impl<'a, T> SubAssign<&'a T> for SignalBase
where
    T: AsPrimitive<f64>,
{
    fn sub_assign(&mut self, other: &'a T) {
        self.base -= other.as_();
    }
}

impl<'a, T> MulAssign<&'a T> for SignalBase
where
    T: AsPrimitive<f64>,
{
    fn mul_assign(&mut self, other: &'a T) {
        self.base *= other.as_();
    }
}

impl<'a, T> DivAssign<&'a T> for SignalBase
where
    T: AsPrimitive<f64>,
{
    fn div_assign(&mut self, other: &'a T) {
        self.base /= other.as_();
    }
}
// 实现差分和反转等操作
impl SignalBase {
    pub fn diff(&self) -> Result<SignalBase, BaseOperationError> {
        let len = self.len();
        if len < 2 {
            return Err(BaseOperationError::DiffShortLength);
        }
        Ok(SignalBase::from_iter(
            self.base.windows(2).into_iter().map(|w| w[1] - w[0]),
        ))
    }

    pub fn rev(&self) -> Self {
        SignalBase::from_iter(self.base.iter().rev().cloned())
    }
}

pub enum PadSide {
    Left,
    Right,
    Both,
}
impl SignalBase {
    fn concat(&self, another: &SignalBase) -> Self {
        SignalBase {
            base: concatenate![Axis(0), self.base, another.base],
        }
    }

    pub fn pad_cons<T>(
        &self,
        pad_side: PadSide,
        constants: T,
        pad_width: usize,
    ) -> Result<Self, PadError>
    where
        T: AsPrimitive<f64>,
    {
        let cons_signal = Self::from_elem(constants.as_(), pad_width);
        match pad_side {
            PadSide::Left => {
                if self.len() + pad_width > isize::MAX as usize {
                    return Err(PadError::ResultOutOfBounds);
                }
                Ok(self.concat(&cons_signal))
            }
            PadSide::Right => {
                if self.len() + pad_width > isize::MAX as usize {
                    return Err(PadError::ResultOutOfBounds);
                }
                Ok(cons_signal.concat(self))
            }
            PadSide::Both => {
                if self.len() + pad_width > isize::MAX as usize {
                    return Err(PadError::ResultOutOfBounds);
                }
                Ok(cons_signal.concat(self).concat(&cons_signal))
            }
        }
    }

    pub fn pad_wrap(&self, pad_side: PadSide, pad_width: usize) -> Result<Self, PadError> {
        let len = self.len();
        if len + 2 * pad_width > isize::MAX as usize {
            return Err(PadError::ResultOutOfBounds);
        }
        if len == 0 {
            return Err(PadError::ConcatenationError("Empty input".to_string()));
        }
        let repeat_count = pad_width / len;
        let remain = pad_width % len;
        let views = repeat(self.base.view())
            .take(repeat_count)
            .collect::<Vec<_>>();
        let repeated = match concatenate(Axis(0), views.as_slice()) {
            Ok(v) => v,
            Err(e) => {
                unreachable!("Unexpected concatenation error for Array1: {}", e);
            }
        };

        let padded = match pad_side {
            PadSide::Left => {
                let left_iter = self
                    .base
                    .as_slice()
                    .unwrap()
                    .iter()
                    .skip(len - remain)
                    .rev();
                left_iter
                    .chain(repeated.iter())
                    .chain(self.base.iter())
                    .cloned()
                    .collect()
            }
            PadSide::Right => {
                let right_iter = self
                    .base
                    .as_slice()
                    .unwrap()
                    .iter()
                    .rev()
                    .skip(len - remain)
                    .rev();
                self.base
                    .iter()
                    .chain(repeated.iter())
                    .chain(right_iter)
                    .cloned()
                    .collect()
            }
            PadSide::Both => {
                let left_iter = self
                    .base
                    .as_slice()
                    .unwrap()
                    .iter()
                    .skip(len - remain)
                    .rev();
                let right_iter = self
                    .base
                    .as_slice()
                    .unwrap()
                    .iter()
                    .rev()
                    .skip(len - remain)
                    .rev();
                left_iter
                    .chain(repeated.iter())
                    .chain(self.base.iter())
                    .chain(repeated.iter())
                    .chain(right_iter)
                    .cloned()
                    .collect()
            }
        };

        Ok(SignalBase { base: padded })
    }
}
