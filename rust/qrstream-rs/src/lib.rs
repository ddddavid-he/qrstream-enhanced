use pyo3::prelude::*;

mod payload;
mod protocol_v4;
mod v4_session;

use v4_session::{DecodeResult, DecodeSnapshot, V4DecodeSession};

#[pymodule]
fn qrstream_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<V4DecodeSession>()?;
    m.add_class::<DecodeResult>()?;
    m.add_class::<DecodeSnapshot>()?;
    Ok(())
}
