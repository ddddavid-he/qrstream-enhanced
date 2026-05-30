#[cfg(feature = "python")]
use pyo3::prelude::*;

mod ffi;
mod payload;
mod protocol_v4;
mod v4_core;
#[cfg(feature = "python")]
mod v4_session;

uniffi::include_scaffolding!("qrstream_core");

use ffi::{FfiDecodeResult, FfiDecodeSnapshot, FfiV4DecodeSession};
#[cfg(feature = "python")]
use v4_session::{DecodeResult, DecodeSnapshot, V4DecodeSession};

#[cfg(feature = "python")]
#[pymodule]
fn qrstream_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<V4DecodeSession>()?;
    m.add_class::<DecodeResult>()?;
    m.add_class::<DecodeSnapshot>()?;
    Ok(())
}
