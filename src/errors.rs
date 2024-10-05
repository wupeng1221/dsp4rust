use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug, Clone)]
pub enum DiffError {
    ShortLength,
}

impl Display for DiffError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffError::ShortLength => {
                write!(
                    f,
                    "Function \'diff\' requires the wrapper length to be at least 2"
                )
            }
        }
    }
}

impl Error for DiffError {}

#[derive(Debug, Clone)]
pub enum PadError {
    EmptyInput,
    UnknownError(String),
}
