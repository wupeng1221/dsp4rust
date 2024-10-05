use crate::inner::base::SignalBase;
use num_traits::AsPrimitive;
use std::fmt::Display;
use std::ops::{Deref, DerefMut, Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};

/// Represents a signal structure for digital signal processing.
///
/// `Signal` encapsulates `SignalBase` and provides a series of methods for creating and manipulating signals.
/// It offers a high-level interface for various signal processing operations.
///
/// # Features
/// - Supports indexing with `isize`, allowing for negative indices (circular indexing).
/// - Implements basic arithmetic operations (+, -, *, /) between signals.
/// - Supports in-place arithmetic operations (+=, -=, *=, /=) for signals.
/// - Supports arithmetic operations with scalars on the right side. These scalars can be any type
///   that implements the `AsPrimitive<f64>` trait.
/// - Provides methods for creating signals from various sources (vectors, iterators, functions).
/// - Offers utility functions like `zeros`, `ones`, `linspace`, and `arrange` for signal generation.
///
/// # Important Notes
/// - All arithmetic operations (including in-place operations) are implemented using reference passing only.
///   This design choice ensures a unified interface and prevents unintended ownership transfers.
/// - Scalar operations are only supported with the scalar on the right side of the operator.
///
/// # Underlying Implementation
/// While the core functionality is implemented in the `SignalBase` struct, `Signal` provides a more
/// user-friendly interface. It delegates most operations to `SignalBase` through deref coercion.
///
/// # Examples
/// ```
/// use dsp4rust::signal::Signal;
///
/// // Create signals
/// let mut signal1 = Signal::from_vec(vec![1.0, 2.0, 3.0]);
/// let signal2 = Signal::from_vec(vec![4.0, 5.0, 6.0]);
///
/// // Arithmetic operations (using references)
/// let sum = &signal1 + &signal2;
/// let product = &signal1 * &2.0;  // Scalar on the right side
/// let product_int = &signal1 * &2;  // Integer scalar also works
///
/// // In-place operations
/// signal1 += &signal2;
/// signal1 *= &2.0;
///
/// // Indexing
/// assert_eq!(signal1[0], 10.0);  // (1.0 + 4.0) * 2.0
/// assert_eq!(signal1[-1], 18.0);  // (3.0 + 6.0) * 2.0 (circular indexing)
/// ```
///
/// 表示用于数字信号处理的信号结构。
///
/// `Signal` 封装了 `SignalBase`，提供了一系列用于创建和操作信号的方法。
/// 它为各种信号处理操作提供了高级接口。
///
/// # 特性
/// - 支持使用 `isize` 进行索引，允许负索引（循环索引）。
/// - 实现了信号之间的基本算术运算（+, -, *, /）。
/// - 支持信号的原地算术运算（+=, -=, *=, /=）。
/// - 支持与标量的算术运算，但标量必须在右。这些标量可以是任何实现了 `AsPrimitive<f64>` trait 的类型。
/// - 提供了从各种源（向量、迭代器、函数）创建信号的方法。
/// - 提供了用于信号生成的实用函数，如 `zeros`、`ones`、`linspace` 和 `arrange`。
///
/// # 重要说明
/// - 所有算术运算（包括原地运算）都只实现了引用传递。
///   这种设计选择确保了统一的接口，并防止了意外的所有权转移。
/// - 标量运算只支持标量在运算符的右侧。
///
/// # 底层实现
/// 虽然核心功能在 `SignalBase` 结构体中实现，但 `Signal` 提供了更加用户友好的接口。
/// 它通过 deref 强制转换将大多数操作委托给 `SignalBase`。
///
/// # 示例
/// ```
/// use dsp4rust::signal::Signal;
///
/// // 创建信号
/// let mut signal1 = Signal::from_vec(vec![1.0, 2.0, 3.0]);
/// let signal2 = Signal::from_vec(vec![4.0, 5.0, 6.0]);
///
/// // 算术运算（使用引用）
/// let sum = &signal1 + &signal2;
/// let product = &signal1 * &2.0;  // 标量在右侧
/// let product_int = &signal1 * &2;  // 整数标量也可以
///
/// // 原地运算
/// signal1 += &signal2;
/// signal1 *= &2.0;
///
/// // 索引
/// assert_eq!(signal1[0], 10.0);  // (1.0 + 4.0) * 2.0
/// assert_eq!(signal1[-1], 18.0);  // (3.0 + 6.0) * 2.0（循环引）
/// ```
#[derive(Debug, Clone)]
pub struct Signal {
    signal: SignalBase,
}

impl Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Signal[len = {}, {}]", self.signal.len(), self.signal)
    }
}

impl<F> FromIterator<F> for Signal
where
    F: AsPrimitive<f64>,
{
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        Signal::from_base(SignalBase::from_iter(iter))
    }
}

impl Deref for Signal {
    type Target = SignalBase;

    fn deref(&self) -> &Self::Target {
        &self.signal
    }
}

impl DerefMut for Signal {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.signal
    }
}

// Static functions
impl Signal {
    /// Creates a `Signal` from `SignalBase`.
    ///
    /// This is an internal method used to create a `Signal` from a `SignalBase` instance.
    ///
    /// 从 `SignalBase` 创建 `Signal`。
    ///
    /// 这是一个内部使用的方法，用于从 `SignalBase` 实例创建 `Signal`。
    fn from_base(signal_base: SignalBase) -> Self {
        Signal {
            signal: signal_base,
        }
    }

    /// Creates a `Signal` from `Vec<f64>`.
    ///
    /// # Parameters
    /// * `vec` - A `Vec<f64>` containing signal data.
    ///
    /// 从 `Vec<f64>` 建 `Signal`。
    ///
    /// # 参数
    /// * `vec` - 包含信号数据的 `Vec<f64>`。
    ///
    /// # Example
    /// ```
    /// use dsp4rust::signal::Signal;
    /// let signal = Signal::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(signal.len(), 3);
    /// ```
    pub fn from_vec(vec: Vec<f64>) -> Self {
        Self::from_base(SignalBase::from_vec(vec))
    }

    /// Creates a `Signal` with a specified element.
    ///
    /// # Parameters
    /// * `ele` - The value to fill the signal with.
    /// * `len` - The length of the signal.
    ///
    /// 创建一个包含指定元素的 `Signal`。
    ///
    /// # 参数
    /// * `ele` - 用于填充信号的值。
    /// * `len` - 信号的长度。
    ///
    /// # Example
    /// ```
    /// use dsp4rust::signal::Signal;
    /// let signal = Signal::from_elem(1.0, 5);
    /// assert_eq!(signal.len(), 5);
    /// assert!(signal.iter().all(|&x| x == 1.0));
    /// ```
    pub fn from_elem(ele: f64, len: usize) -> Self {
        Self::from_base(SignalBase::from_elem(ele, len))
    }

    /// Creates a `Signal` using a given function.
    ///
    /// # Parameters
    /// * `len` - The length of the signal.
    /// * `f` - A function that takes an index and returns the corresponding signal value.
    ///
    /// 使用给定的函创建 `Signal`。
    ///
    /// # 参数
    /// * `len` - 信号的长度。
    /// * `f` - 一个函数，接受索引并返回对应的信号值。
    ///
    /// # Example
    /// ```
    /// use dsp4rust::signal::Signal;
    /// let signal = Signal::from_len_fn(5, |i| i as f64);
    /// assert_eq!(signal.len(), 5);
    /// assert_eq!(signal[3], 3.0);
    /// ```
    pub fn from_len_fn<F>(len: usize, f: F) -> Self
    where
        F: Fn(usize) -> f64,
    {
        Self::from_base(SignalBase::from_len_fn(len, f))
    }

    /// Creates a `Signal` from an iterable object.
    ///
    /// # Parameters
    /// * `iter` - An iterable object whose elements can be converted to `f64`.
    ///
    /// 从可迭代对象创建 `Signal`。
    ///
    /// # 参数
    /// * `iter` - 一个可迭代对象，其元素可以转换为 `f64`。
    ///
    /// # Example
    /// ```
    /// use dsp4rust::signal::Signal;
    /// let signal = Signal::from_iter(vec![1, 2, 3]);
    /// assert_eq!(signal.len(), 3);
    /// assert_eq!(signal[1], 2.0);
    /// ```
    pub fn from_iter<I, T>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: AsPrimitive<f64>,
    {
        Self::from_base(SignalBase::from_iter(
            iter.into_iter().map(AsPrimitive::as_),
        ))
    }

    /// Creates a zero signal of specified length.
    ///
    /// # Parameters
    /// * `len` - The length of the signal.
    ///
    /// 创建一个指定长度的全零信号。
    ///
    /// # 参数
    /// * `len` - 信号的长度。
    ///
    /// # Example
    /// ```
    /// use dsp4rust::signal::Signal;
    /// let signal = Signal::zeros(3);
    /// assert_eq!(signal.len(), 3);
    /// assert!(signal.iter().all(|&x| x == 0.0));
    /// ```
    pub fn zeros(len: usize) -> Self {
        Self::from_base(SignalBase::zeros(len))
    }

    /// Creates a signal of ones with specified length.
    ///
    /// # Parameters
    /// * `len` - The length of the signal.
    ///
    /// 创建一个指定长度的全一信号。
    ///
    /// # 参数
    /// * `len` - 信号的长度。
    ///
    /// # Example
    /// ```
    /// use dsp4rust::signal::Signal;
    /// let signal = Signal::ones(3);
    /// assert_eq!(signal.len(), 3);
    /// assert!(signal.iter().all(|&x| x == 1.0));
    /// ```
    pub fn ones(len: usize) -> Self {
        Self::from_base(SignalBase::ones(len))
    }

    /// Creates a linearly spaced signal.
    ///
    /// # Parameters
    /// * `start` - The starting value.
    /// * `end` - The ending value (inclusive).
    /// * `len` - The length of the signal.
    ///
    /// 创建一个线性间隔的信号。
    ///
    /// # 参数
    /// * `start` - 起始值。
    /// * `end` - 结束值（包含）。
    /// * `len` - 信号的长度。
    ///
    /// # Example
    /// ```
    /// use dsp4rust::signal::Signal;
    /// let signal = Signal::linspace(0.0, 1.0, 3);
    /// assert_eq!(signal.len(), 3);
    /// assert_eq!(signal[0], 0.0);
    /// assert_eq!(signal[2], 1.0);
    /// ```
    pub fn linspace(start: f64, end: f64, len: usize) -> Self {
        Self::from_base(SignalBase::linspace(start, end, len))
    }

    /// Creates an arithmetic sequence signal.
    ///
    /// # Parameters
    /// * `start` - The starting value.
    /// * `end_exclude` - The ending value (exclusive).
    /// * `step` - The step size.
    ///
    /// # Note
    /// The generated signal does not include the `end_exclude` value.
    ///
    /// 创建一个等差数列信号。
    ///
    /// # 参数
    /// * `start` - 起始值。
    /// * `end_exclude` - 结束值（不包含）。
    /// * `step` - 步长。
    ///
    /// # 注意
    /// 生成的信号不包含 `end_exclude` 值。
    ///
    /// # Example
    /// ```
    /// use dsp4rust::signal::Signal;
    /// let signal = Signal::arrange(0.0, 5.0, 2.0);
    /// assert_eq!(signal.len(), 3);
    /// assert_eq!(signal[0], 0.0);
    /// assert_eq!(signal[2], 4.0);
    /// ```
    pub fn arrange(start: f64, end_exclude: f64, step: f64) -> Self {
        Self::from_base(SignalBase::arrange(start, end_exclude, step))
    }
}

// 实现 Signal 与 Signal 的运算
impl<'a, 'b> Add<&'b Signal> for &'a Signal {
    type Output = Signal;
    fn add(self, other: &'b Signal) -> Signal {
        Signal::from_base(&self.signal + &other.signal)
    }
}

impl<'a, 'b> Sub<&'b Signal> for &'a Signal {
    type Output = Signal;
    fn sub(self, other: &'b Signal) -> Signal {
        Signal::from_base(&self.signal - &other.signal)
    }
}

impl<'a, 'b> Mul<&'b Signal> for &'a Signal {
    type Output = Signal;
    fn mul(self, other: &'b Signal) -> Signal {
        Signal::from_base(&self.signal * &other.signal)
    }
}

impl<'a, 'b> Div<&'b Signal> for &'a Signal {
    type Output = Signal;
    fn div(self, other: &'b Signal) -> Signal {
        Signal::from_base(&self.signal / &other.signal)
    }
}

// 实现 Signal 的原地运算
impl<'a> AddAssign<&'a Signal> for Signal {
    fn add_assign(&mut self, other: &'a Signal) {
        self.signal += &other.signal;
    }
}

impl<'a> SubAssign<&'a Signal> for Signal {
    fn sub_assign(&mut self, other: &'a Signal) {
        self.signal -= &other.signal;
    }
}

impl<'a> MulAssign<&'a Signal> for Signal {
    fn mul_assign(&mut self, other: &'a Signal) {
        self.signal *= &other.signal;
    }
}

impl<'a> DivAssign<&'a Signal> for Signal {
    fn div_assign(&mut self, other: &'a Signal) {
        self.signal /= &other.signal;
    }
}

// 实现 Signal 与标量的运算
impl<'a, T> Add<&'a T> for &'a Signal 
where 
    T: AsPrimitive<f64> 
{
    type Output = Signal;
    fn add(self, other: &'a T) -> Signal {
        Signal::from_base(&self.signal + other)
    }
}

impl<'a, T> Sub<&'a T> for &'a Signal 
where 
    T: AsPrimitive<f64> 
{
    type Output = Signal;
    fn sub(self, other: &'a T) -> Signal {
        Signal::from_base(&self.signal - other)
    }
}

impl<'a, T> Mul<&'a T> for &'a Signal 
where 
    T: AsPrimitive<f64> 
{
    type Output = Signal;
    fn mul(self, other: &'a T) -> Signal {
        Signal::from_base(&self.signal * other)
    }
}

impl<'a, T> Div<&'a T> for &'a Signal 
where 
    T: AsPrimitive<f64> 
{
    type Output = Signal;
    fn div(self, other: &'a T) -> Signal {
        Signal::from_base(&self.signal / other)
    }
}

// 为 Signal 实现与引用标量的原地运算
impl<'a, T> AddAssign<&'a T> for Signal 
where 
    T: AsPrimitive<f64> 
{
    fn add_assign(&mut self, other: &'a T) {
        self.signal += other;
    }
}

impl<'a, T> SubAssign<&'a T> for Signal 
where 
    T: AsPrimitive<f64> 
{
    fn sub_assign(&mut self, other: &'a T) {
        self.signal -= other;
    }
}

impl<'a, T> MulAssign<&'a T> for Signal 
where 
    T: AsPrimitive<f64> 
{
    fn mul_assign(&mut self, other: &'a T) {
        self.signal *= other;
    }
}

impl<'a, T> DivAssign<&'a T> for Signal 
where 
    T: AsPrimitive<f64> 
{
    fn div_assign(&mut self, other: &'a T) {
        self.signal /= other;
    }
}

/// Creates a Signal from a list of values or repeated value.
///
/// This macro provides two ways to create a Signal:
/// 1. From a list of values
/// 2. From a single value repeated a specified number of times
///
/// 从值列表或重复值创建信号。
///
/// 这个宏提供了两种创建信号的方式：
/// 1. 从值列表创建
/// 2. 从单个值重复指定次数创建
///
/// # Examples
///
/// Creating a Signal from a list of values:
///
/// 从值列表创建信号：
///
/// ```
/// use dsp4rust::signal;
///
/// let sig = signal![1.0, 2.0, 3.0, 4.0];
/// ```
///
/// Creating a Signal from a repeated value:
///
/// 从重复值创建信号：
///
/// ```
/// use dsp4rust::signal;
///
/// let sig = signal![1.0; 5];
/// ```
#[macro_export]
macro_rules! signal {
    ($($x:expr),* $(,)?) => {
        {
            let vec = vec![$($x as f64),*];
            $crate::signal::Signal::from_vec(vec)
        }
    };
    ($x:expr; $n:expr) => {
        {
            $crate::signal::Signal::from_elem($x as f64, $n)
        }
    };
}