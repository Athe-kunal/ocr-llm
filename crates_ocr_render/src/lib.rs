use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use image::{DynamicImage, ImageFormat};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::io::Cursor;
use std::process::Command;

/// Render a single PDF page to a base64-encoded PNG string.
///
/// Invokes `pdftoppm` for rasterisation (same as the Python implementation) but
/// performs the base64 encoding in Rust, which is significantly faster than
/// Python's `base64.b64encode` for large payloads and releases the GIL while
/// running so other async tasks are not blocked.
///
/// Args:
///     pdf_path: Absolute path to the PDF file.
///     page_num: 1-based page number to render.
///     target_longest_dim: Target size (pixels) for the longest dimension.
///
/// Returns:
///     Base64-encoded PNG bytes as a Python `str`.
#[pyfunction]
fn render_pdf_page_to_base64png(
    pdf_path: &str,
    page_num: u32,
    target_longest_dim: u32,
) -> PyResult<String> {
    let output = Command::new("pdftoppm")
        .args([
            "-png",
            "-f",
            &page_num.to_string(),
            "-l",
            &page_num.to_string(),
            "-scale-to",
            &target_longest_dim.to_string(),
            pdf_path,
        ])
        .output()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to spawn pdftoppm: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PyRuntimeError::new_err(format!(
            "pdftoppm failed (exit {}): {stderr}",
            output.status
        )));
    }

    Ok(B64.encode(&output.stdout))
}

/// Rotate a base64-encoded PNG image by the given number of degrees.
///
/// Replaces the PIL-based rotation in `build_page_query` which runs
/// synchronously in the asyncio event loop. This function is called via
/// `asyncio.to_thread` from Python so the GIL is released for the duration.
///
/// Args:
///     base64_png: Base64-encoded PNG image.
///     rotation_degrees: Clockwise rotation to apply. Must be 90, 180, or 270.
///
/// Returns:
///     Base64-encoded PNG bytes of the rotated image as a Python `str`.
#[pyfunction]
fn rotate_base64_png(base64_png: &str, rotation_degrees: u32) -> PyResult<String> {
    let png_bytes = B64
        .decode(base64_png)
        .map_err(|e| PyRuntimeError::new_err(format!("base64 decode failed: {e}")))?;

    let img = image::load_from_memory_with_format(&png_bytes, ImageFormat::Png)
        .map_err(|e| PyRuntimeError::new_err(format!("PNG decode failed: {e}")))?;

    let rotated: DynamicImage = match rotation_degrees {
        90 => DynamicImage::from(image::imageops::rotate90(&img)),
        180 => DynamicImage::from(image::imageops::rotate180(&img)),
        270 => DynamicImage::from(image::imageops::rotate270(&img)),
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported rotation {other}; must be 90, 180, or 270"
            )));
        }
    };

    let mut buf = Cursor::new(Vec::new());
    rotated
        .write_to(&mut buf, ImageFormat::Png)
        .map_err(|e| PyRuntimeError::new_err(format!("PNG encode failed: {e}")))?;

    Ok(B64.encode(buf.into_inner()))
}

#[pymodule]
fn ocr_render_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_pdf_page_to_base64png, m)?)?;
    m.add_function(wrap_pyfunction!(rotate_base64_png, m)?)?;
    Ok(())
}
