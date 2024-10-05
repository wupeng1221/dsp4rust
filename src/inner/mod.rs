pub mod base;

use thiserror::Error;

/// Errors that can occur during basic operations on SignalBase.
///
/// SignalBase 基本操作过程中可能发生的错误。
#[derive(Error, Debug)]
pub enum BaseOperationError {
    /// Error when trying to compute the difference of a signal that's too short.
    ///
    /// 当尝试计算过短信号的差分时发生的错误。
    #[error("Signal length is too short for differentiation")]
    DiffShortLength,
}

/// Errors that can occur during padding operations on SignalBase.
///
/// 当在 SignalBase 上进行填充操作时可能发生的错误。
#[derive(Error, Debug, Clone, PartialEq)]
pub enum PadError {
    /// Error that occurs during signal concatenation.
    #[error("Concatenation error: {0}")]
    ConcatenationError(String),

    /// Error that occurs when the padding operation results in a value out of representable range.
    #[error("Padding result out of bounds")]
    ResultOutOfBounds,

    /// Other unexpected errors.
    #[error("Other error: {0}")]
    Other(String),
}
