//! Structural verification of model files on disk.
//!
//! This answers one question: is this file what it claims to be, and is all
//! of it here? That covers the failure people actually hit — a download that
//! stopped short, or a file whose header disagrees with its contents — and it
//! is also the check that keeps a parser safe, because both a truncated file
//! and a hostile one look the same to a reader that trusts the header.
//!
//! Every count and length in these formats is attacker-controlled, so nothing
//! here allocates on a number the file supplied without first proving the
//! bytes exist to back it. A file claiming two billion tensors is rejected in
//! constant time rather than exhausting memory.
//!
//! What this is not: it does not execute anything, does not inspect weights
//! for behaviour, and cannot tell a well-formed malicious model from a
//! well-formed honest one. It says the container is intact.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use serde::Serialize;

/// The largest header string or JSON blob worth reading. Both formats cap out
/// far below this in practice; the limit exists so a corrupt length field
/// fails fast instead of reserving whatever it asked for.
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

/// Smallest possible on-disk size of one tensor descriptor (an empty name,
/// zero dimensions, type and offset). Used to reject an impossible tensor
/// count before the loop that would otherwise run it.
const MIN_TENSOR_INFO_BYTES: u64 = 8 + 4 + 4 + 8;

/// Smallest possible size of one metadata entry: an empty key and a type tag.
const MIN_METADATA_BYTES: u64 = 8 + 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Gguf,
    Safetensors,
}

impl Format {
    pub fn label(self) -> &'static str {
        match self {
            Format::Gguf => "GGUF",
            Format::Safetensors => "safetensors",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Note,
    Warning,
    Error,
}

impl Level {
    pub fn label(self) -> &'static str {
        match self {
            Level::Note => "note",
            Level::Warning => "warning",
            Level::Error => "error",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub level: Level,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub path: String,
    pub format: Format,
    pub file_size: u64,
    pub version: Option<u32>,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub context_length: Option<u64>,
    /// Summed from the tensor shapes, not read from metadata: this is what
    /// the file actually contains rather than what it says it contains.
    pub parameters: u64,
    /// Tensor element types and how many tensors use each, largest first.
    pub tensor_types: Vec<TypeCount>,
    /// Bytes the tensor data occupies according to the header.
    pub data_bytes: u64,
    pub findings: Vec<Finding>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TypeCount {
    pub name: String,
    pub tensors: usize,
}

impl Report {
    /// True when nothing structural is wrong. Warnings do not count: an
    /// unrecognised tensor type is a gap in this tool's table, not damage to
    /// the file.
    pub fn is_intact(&self) -> bool {
        !self.findings.iter().any(|f| f.level == Level::Error)
    }

    fn note(&mut self, message: impl Into<String>) {
        self.findings.push(Finding {
            level: Level::Note,
            message: message.into(),
        });
    }

    fn warn(&mut self, message: impl Into<String>) {
        self.findings.push(Finding {
            level: Level::Warning,
            message: message.into(),
        });
    }

    fn error(&mut self, message: impl Into<String>) {
        self.findings.push(Finding {
            level: Level::Error,
            message: message.into(),
        });
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn verify(path: &Path) -> Result<Report, String> {
    let display = path.display().to_string();
    let file = File::open(path).map_err(|e| format!("cannot open {display}: {e}"))?;
    let file_size = file
        .metadata()
        .map_err(|e| format!("cannot stat {display}: {e}"))?
        .len();
    let mut reader = Cursor::new(BufReader::new(file), file_size);

    match detect(&mut reader)? {
        Format::Gguf => gguf(&mut reader, display, file_size),
        Format::Safetensors => safetensors(&mut reader, display, file_size),
    }
}

/// Identify the format from its opening bytes rather than the file name — an
/// extension is a claim, and this whole module exists to check claims.
fn detect(reader: &mut Cursor) -> Result<Format, String> {
    let mut magic = [0u8; 8];
    reader
        .read_exact_at(0, &mut magic)
        .map_err(|_| "file is too small to be a model file".to_string())?;
    if &magic[..4] == b"GGUF" {
        return Ok(Format::Gguf);
    }
    // safetensors opens with the JSON header's length, so the first byte of
    // that header is the one just past it.
    let mut brace = [0u8; 1];
    if reader.read_exact_at(8, &mut brace).is_ok() && brace[0] == b'{' {
        return Ok(Format::Safetensors);
    }
    Err("not a GGUF or safetensors file".to_string())
}

// ---------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------

fn gguf(reader: &mut Cursor, path: String, file_size: u64) -> Result<Report, String> {
    reader.seek_to(4)?; // past the magic
    let version = reader.u32()?;
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;

    let mut report = Report {
        path,
        format: Format::Gguf,
        file_size,
        version: Some(version),
        tensor_count,
        metadata_count,
        architecture: None,
        name: None,
        context_length: None,
        parameters: 0,
        tensor_types: Vec::new(),
        data_bytes: 0,
        findings: Vec::new(),
    };

    if !(1..=3).contains(&version) {
        report.warn(format!(
            "GGUF version {version} is newer than this build understands (1-3); \
             reading it as version 3"
        ));
    }

    // Reject impossible counts before the loops that would honour them.
    let remaining = file_size.saturating_sub(reader.pos());
    if metadata_count.saturating_mul(MIN_METADATA_BYTES) > remaining {
        report.error(format!(
            "header claims {metadata_count} metadata entries, which cannot fit in the \
             {remaining} bytes that follow it"
        ));
        return Ok(report);
    }
    if tensor_count.saturating_mul(MIN_TENSOR_INFO_BYTES) > remaining {
        report.error(format!(
            "header claims {tensor_count} tensors, which cannot fit in the {remaining} \
             bytes that follow it"
        ));
        return Ok(report);
    }

    let metadata = match read_metadata(reader, metadata_count) {
        Ok(metadata) => metadata,
        Err(e) => {
            report.error(format!("metadata is malformed: {e}"));
            return Ok(report);
        }
    };

    report.architecture = metadata.get("general.architecture").cloned();
    report.name = metadata.get("general.name").cloned();
    if let Some(arch) = &report.architecture {
        report.context_length = metadata
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.parse::<u64>().ok());
    }
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(32);
    if !alignment.is_power_of_two() {
        report.error(format!(
            "general.alignment is {alignment}, which is not a power of two"
        ));
        return Ok(report);
    }

    let tensors = match read_tensor_infos(reader, tensor_count) {
        Ok(tensors) => tensors,
        Err(e) => {
            report.error(format!("tensor table is malformed: {e}"));
            return Ok(report);
        }
    };

    // Tensor data starts at the first alignment boundary after the header.
    let header_end = reader.pos();
    let data_start = header_end.div_ceil(alignment) * alignment;
    if data_start > file_size {
        report.error(format!(
            "the header runs to {header_end} bytes, past the end of a {file_size}-byte file"
        ));
        return Ok(report);
    }

    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut unknown_types: BTreeMap<u32, usize> = BTreeMap::new();
    let mut data_end = 0u64;

    for tensor in &tensors {
        let elements: u64 = tensor
            .dims
            .iter()
            .copied()
            .try_fold(1u64, |acc, d| acc.checked_mul(d))
            .unwrap_or(u64::MAX);
        if elements == u64::MAX {
            report.error(format!(
                "tensor '{}' declares dimensions that overflow a 64-bit element count",
                tensor.name
            ));
            continue;
        }
        report.parameters = report.parameters.saturating_add(elements);

        let Some(kind) = GgmlType::from_id(tensor.kind) else {
            *unknown_types.entry(tensor.kind).or_insert(0) += 1;
            continue;
        };
        *counts.entry(kind.name).or_insert(0) += 1;

        if elements % kind.block != 0 {
            report.warn(format!(
                "tensor '{}' has {elements} elements, not a multiple of {}'s block size {}",
                tensor.name, kind.name, kind.block
            ));
        }
        let bytes = (elements / kind.block).saturating_mul(kind.size);
        let end = match data_start
            .checked_add(tensor.offset)
            .and_then(|start| start.checked_add(bytes))
        {
            Some(end) => end,
            None => {
                report.error(format!(
                    "tensor '{}' has an offset that overflows the file",
                    tensor.name
                ));
                continue;
            }
        };
        data_end = data_end.max(end);
    }

    for (id, n) in unknown_types {
        report.warn(format!(
            "{n} tensor(s) use element type {id}, which this build does not know; \
             their size could not be checked"
        ));
    }

    report.data_bytes = data_end.saturating_sub(data_start);
    report.tensor_types = counts
        .into_iter()
        .map(|(name, tensors)| TypeCount {
            name: name.to_string(),
            tensors,
        })
        .collect();
    report
        .tensor_types
        .sort_by(|a, b| b.tensors.cmp(&a.tensors).then(a.name.cmp(&b.name)));

    // The check this command exists for.
    if data_end > file_size {
        let missing = data_end - file_size;
        report.error(format!(
            "the file is {} short: the tensor table describes {} of data but the file \
             ends at {}. The download did not finish.",
            human(missing),
            human(data_end),
            human(file_size)
        ));
    } else if file_size > data_end && file_size - data_end > alignment {
        report.note(format!(
            "{} of trailing bytes after the last tensor",
            human(file_size - data_end)
        ));
    }

    if tensor_count == 0 {
        report.warn("the file declares no tensors");
    }

    Ok(report)
}

struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    kind: u32,
    offset: u64,
}

fn read_metadata(reader: &mut Cursor, count: u64) -> Result<BTreeMap<String, String>, String> {
    let mut map = BTreeMap::new();
    for _ in 0..count {
        let key = reader.string()?;
        let kind = reader.u32()?;
        let value = read_value(reader, kind, 0)?;
        map.insert(key, value);
    }
    Ok(map)
}

/// Read one metadata value, returning a short rendering of it.
///
/// Arrays are counted and discarded rather than collected: a tokenizer
/// vocabulary is a single array with hundreds of thousands of strings in it,
/// and keeping them would cost more memory than the rest of this command
/// combined for a value nothing reads.
fn read_value(reader: &mut Cursor, kind: u32, depth: u32) -> Result<String, String> {
    // Arrays nest in principle; in practice they do not, and a bound here
    // means a crafted file cannot drive this into a stack overflow.
    if depth > 2 {
        return Err("metadata arrays are nested too deeply".to_string());
    }
    Ok(match kind {
        0 => reader.u8()?.to_string(),
        1 => (reader.u8()? as i8).to_string(),
        2 => reader.u16()?.to_string(),
        3 => (reader.u16()? as i16).to_string(),
        4 => reader.u32()?.to_string(),
        5 => (reader.u32()? as i32).to_string(),
        6 => f32::from_bits(reader.u32()?).to_string(),
        7 => (reader.u8()? != 0).to_string(),
        8 => reader.string()?,
        9 => {
            let inner = reader.u32()?;
            let len = reader.u64()?;
            // Same bound as the top-level counts: an array cannot have more
            // entries than the file has bytes left to hold them.
            if len > reader.remaining() {
                return Err(format!(
                    "an array claims {len} entries, more than the file has bytes left"
                ));
            }
            for _ in 0..len {
                read_value(reader, inner, depth + 1)?;
            }
            format!("[{len} values]")
        }
        10 => reader.u64()?.to_string(),
        11 => (reader.u64()? as i64).to_string(),
        12 => f64::from_bits(reader.u64()?).to_string(),
        other => return Err(format!("unknown metadata value type {other}")),
    })
}

fn read_tensor_infos(reader: &mut Cursor, count: u64) -> Result<Vec<TensorInfo>, String> {
    let mut tensors = Vec::new();
    for _ in 0..count {
        let name = reader.string()?;
        let n_dims = reader.u32()?;
        // GGUF allows at most four dimensions; anything else is corruption,
        // and honouring it would size the allocation below from the file.
        if n_dims > 4 {
            return Err(format!(
                "tensor '{name}' declares {n_dims} dimensions (max 4)"
            ));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(reader.u64()?);
        }
        let kind = reader.u32()?;
        let offset = reader.u64()?;
        tensors.push(TensorInfo {
            name,
            dims,
            kind,
            offset,
        });
    }
    Ok(tensors)
}

/// The ggml element types, with the block structure their sizes depend on.
///
/// A k-quant stores 256 weights in one block with shared scales, so its size
/// is per block and not per element — which is why this table exists rather
/// than a bytes-per-weight number.
struct GgmlType {
    name: &'static str,
    block: u64,
    size: u64,
}

impl GgmlType {
    fn from_id(id: u32) -> Option<GgmlType> {
        let (name, block, size) = match id {
            0 => ("F32", 1, 4),
            1 => ("F16", 1, 2),
            2 => ("Q4_0", 32, 18),
            3 => ("Q4_1", 32, 20),
            6 => ("Q5_0", 32, 22),
            7 => ("Q5_1", 32, 24),
            8 => ("Q8_0", 32, 34),
            9 => ("Q8_1", 32, 40),
            10 => ("Q2_K", 256, 84),
            11 => ("Q3_K", 256, 110),
            12 => ("Q4_K", 256, 144),
            13 => ("Q5_K", 256, 176),
            14 => ("Q6_K", 256, 210),
            15 => ("Q8_K", 256, 292),
            16 => ("IQ2_XXS", 256, 66),
            17 => ("IQ2_XS", 256, 74),
            18 => ("IQ3_XXS", 256, 98),
            19 => ("IQ1_S", 256, 50),
            20 => ("IQ4_NL", 32, 18),
            21 => ("IQ3_S", 256, 110),
            22 => ("IQ2_S", 256, 82),
            23 => ("IQ4_XS", 256, 136),
            24 => ("I8", 1, 1),
            25 => ("I16", 1, 2),
            26 => ("I32", 1, 4),
            27 => ("I64", 1, 8),
            28 => ("F64", 1, 8),
            29 => ("IQ1_M", 256, 56),
            30 => ("BF16", 1, 2),
            _ => return None,
        };
        Some(GgmlType { name, block, size })
    }
}

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------

fn safetensors(reader: &mut Cursor, path: String, file_size: u64) -> Result<Report, String> {
    reader.seek_to(0)?;
    let header_len = reader.u64()?;

    let mut report = Report {
        path,
        format: Format::Safetensors,
        file_size,
        version: None,
        tensor_count: 0,
        metadata_count: 0,
        architecture: None,
        name: None,
        context_length: None,
        parameters: 0,
        tensor_types: Vec::new(),
        data_bytes: 0,
        findings: Vec::new(),
    };

    if header_len > MAX_HEADER_BYTES || header_len > file_size.saturating_sub(8) {
        report.error(format!(
            "the header claims to be {} long, which a {}-byte file cannot hold",
            human(header_len),
            file_size
        ));
        return Ok(report);
    }

    let mut bytes = vec![0u8; header_len as usize];
    if reader.read_exact_at(8, &mut bytes).is_err() {
        report.error("the header is shorter than it claims — the file is truncated");
        return Ok(report);
    }
    let header: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(e) => {
            report.error(format!("the header is not valid JSON: {e}"));
            return Ok(report);
        }
    };
    let Some(entries) = header.as_object() else {
        report.error("the header is not a JSON object");
        return Ok(report);
    };

    let data_start = 8 + header_len;
    let data_len = file_size - data_start;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    // Kept to prove the tensors tile the data region without overlapping.
    let mut spans: Vec<(u64, u64, String)> = Vec::new();

    for (name, entry) in entries {
        if name == "__metadata__" {
            report.metadata_count = entry.as_object().map(|m| m.len() as u64).unwrap_or(0);
            continue;
        }
        report.tensor_count += 1;

        let dtype = entry.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
        let shape: Vec<u64> = entry
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();
        let offsets: Vec<u64> = entry
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();

        if offsets.len() != 2 {
            report.error(format!("tensor '{name}' has no usable data_offsets"));
            continue;
        }
        let (begin, end) = (offsets[0], offsets[1]);
        if begin > end {
            report.error(format!(
                "tensor '{name}' starts at {begin} and ends at {end}"
            ));
            continue;
        }
        if end > data_len {
            report.error(format!(
                "tensor '{name}' ends at {end} of the data region, which is only {data_len} bytes"
            ));
            continue;
        }

        let elements = shape
            .iter()
            .copied()
            .try_fold(1u64, |a, d| a.checked_mul(d));
        match (elements, dtype_size(dtype)) {
            (Some(elements), Some(width)) => {
                let expected = elements.saturating_mul(width);
                if expected != end - begin {
                    report.error(format!(
                        "tensor '{name}' is {dtype}{shape:?}, which needs {expected} bytes, \
                         but its offsets span {}",
                        end - begin
                    ));
                }
                report.parameters = report.parameters.saturating_add(elements);
            }
            (None, _) => report.error(format!("tensor '{name}' has an overflowing shape")),
            (_, None) => report.warn(format!(
                "tensor '{name}' uses dtype '{dtype}', which this build does not know; \
                 its size could not be checked"
            )),
        }

        *counts.entry(dtype.to_string()).or_insert(0) += 1;
        spans.push((begin, end, name.clone()));
        report.data_bytes = report.data_bytes.max(end);
    }

    // Overlapping tensors are the one corruption the per-tensor checks miss:
    // each span can be individually valid while two of them claim the same
    // bytes.
    spans.sort_by_key(|(begin, _, _)| *begin);
    for pair in spans.windows(2) {
        let (_, first_end, first_name) = &pair[0];
        let (second_begin, _, second_name) = &pair[1];
        if second_begin < first_end {
            report.error(format!(
                "tensors '{first_name}' and '{second_name}' overlap in the data region"
            ));
        }
    }

    if report.data_bytes < data_len {
        report.note(format!(
            "{} of the data region is not claimed by any tensor",
            human(data_len - report.data_bytes)
        ));
    }
    if report.tensor_count == 0 {
        report.warn("the header describes no tensors");
    }

    report.tensor_types = counts
        .into_iter()
        .map(|(name, tensors)| TypeCount { name, tensors })
        .collect();
    report
        .tensor_types
        .sort_by(|a, b| b.tensors.cmp(&a.tensors).then(a.name.cmp(&b.name)));

    Ok(report)
}

fn dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "BOOL" => 1,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// A bounds-checked reader
// ---------------------------------------------------------------------------

/// Sequential reader that knows how big the file is.
///
/// Every read is checked against the real end of the file, so a length taken
/// from the header can never make the parser reach past what exists.
struct Cursor {
    inner: BufReader<File>,
    len: u64,
    pos: u64,
}

impl Cursor {
    fn new(inner: BufReader<File>, len: u64) -> Cursor {
        Cursor { inner, len, pos: 0 }
    }

    fn pos(&self) -> u64 {
        self.pos
    }

    fn remaining(&self) -> u64 {
        self.len.saturating_sub(self.pos)
    }

    fn seek_to(&mut self, pos: u64) -> Result<(), String> {
        self.inner
            .seek(SeekFrom::Start(pos))
            .map_err(|e| format!("cannot seek to {pos}: {e}"))?;
        self.pos = pos;
        Ok(())
    }

    fn read_exact_at(&mut self, pos: u64, buffer: &mut [u8]) -> Result<(), String> {
        self.seek_to(pos)?;
        self.take(buffer)
    }

    fn take(&mut self, buffer: &mut [u8]) -> Result<(), String> {
        let wanted = buffer.len() as u64;
        if wanted > self.remaining() {
            return Err(format!(
                "wanted {wanted} bytes at offset {} but only {} remain",
                self.pos,
                self.remaining()
            ));
        }
        self.inner
            .read_exact(buffer)
            .map_err(|e| format!("read failed at offset {}: {e}", self.pos))?;
        self.pos += wanted;
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, String> {
        let mut b = [0u8; 1];
        self.take(&mut b)?;
        Ok(b[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        let mut b = [0u8; 2];
        self.take(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.take(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let mut b = [0u8; 8];
        self.take(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// A length-prefixed UTF-8 string. The length is checked against the file
    /// before a byte is allocated, which is the whole point of this type.
    fn string(&mut self) -> Result<String, String> {
        let len = self.u64()?;
        if len > self.remaining() || len > MAX_HEADER_BYTES {
            return Err(format!(
                "a string claims to be {len} bytes, more than the file can hold"
            ));
        }
        let mut bytes = vec![0u8; len as usize];
        self.take(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| "a header string is not valid UTF-8".to_string())
    }
}

pub fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("llmspec-verify-{}-{name}", std::process::id()));
        path
    }

    fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = temp(name);
        let mut file = File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    // -- GGUF builders ------------------------------------------------------

    fn gguf_string(s: &str) -> Vec<u8> {
        let mut out = (s.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(s.as_bytes());
        out
    }

    /// A minimal but real GGUF: one metadata string, one Q4_K tensor, and
    /// exactly as many data bytes as the tensor table promises.
    fn good_gguf() -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        out.extend_from_slice(&1u64.to_le_bytes()); // metadata count

        out.extend(gguf_string("general.architecture"));
        out.extend_from_slice(&8u32.to_le_bytes()); // string
        out.extend(gguf_string("llama"));

        out.extend(gguf_string("blk.0.attn_q.weight"));
        out.extend_from_slice(&2u32.to_le_bytes()); // 2 dimensions
        out.extend_from_slice(&256u64.to_le_bytes());
        out.extend_from_slice(&4u64.to_le_bytes());
        out.extend_from_slice(&12u32.to_le_bytes()); // Q4_K
        out.extend_from_slice(&0u64.to_le_bytes()); // offset

        // 1024 elements / 256 per block * 144 bytes = 576 bytes of data.
        let data_start = out.len().div_ceil(32) * 32;
        out.resize(data_start + 576, 0);
        out
    }

    #[test]
    fn a_well_formed_gguf_reads_clean() {
        let path = write("good.gguf", &good_gguf());
        let report = verify(&path).unwrap();
        assert!(report.is_intact(), "findings: {:?}", report.findings);
        assert_eq!(report.format, Format::Gguf);
        assert_eq!(report.version, Some(3));
        assert_eq!(report.tensor_count, 1);
        assert_eq!(report.architecture.as_deref(), Some("llama"));
        assert_eq!(report.parameters, 1024);
        assert_eq!(report.tensor_types[0].name, "Q4_K");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_truncated_download_is_the_error_this_command_exists_for() {
        let mut bytes = good_gguf();
        bytes.truncate(bytes.len() - 200);
        let path = write("short.gguf", &bytes);
        let report = verify(&path).unwrap();
        assert!(!report.is_intact());
        let message = &report
            .findings
            .iter()
            .find(|f| f.level == Level::Error)
            .unwrap()
            .message;
        assert!(
            message.contains("did not finish"),
            "unhelpful message: {message}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_absurd_tensor_count_is_refused_without_allocating_for_it() {
        // The whole safety argument: a hostile header names a number no file
        // could back, and the answer has to be immediate rather than an
        // allocation of that size.
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // tensors
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let path = write("huge.gguf", &bytes);
        let report = verify(&path).unwrap();
        assert!(!report.is_intact());
        assert!(
            report.findings[0].message.contains("cannot fit"),
            "{:?}",
            report.findings
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_absurd_metadata_count_is_refused_too() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // metadata
        let path = write("hugekv.gguf", &bytes);
        let report = verify(&path).unwrap();
        assert!(!report.is_intact());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_string_longer_than_the_file_does_not_allocate_it() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensors
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata
        // A key claiming to be 2 GB in a file of a few dozen bytes.
        bytes.extend_from_slice(&(2u64 << 30).to_le_bytes());
        bytes.extend_from_slice(b"short");
        let path = write("longstring.gguf", &bytes);
        let report = verify(&path).unwrap();
        assert!(!report.is_intact());
        assert!(
            report.findings[0].message.contains("malformed"),
            "{:?}",
            report.findings
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn too_many_dimensions_is_rejected_rather_than_sized_from_the_file() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend(gguf_string("t"));
        bytes.extend_from_slice(&9999u32.to_le_bytes()); // dimensions
        bytes.resize(bytes.len() + 64, 0);
        let path = write("dims.gguf", &bytes);
        let report = verify(&path).unwrap();
        assert!(!report.is_intact());
        assert!(report.findings[0].message.contains("dimensions"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unknown_tensor_type_is_a_warning_not_a_verdict_on_the_file() {
        // A gap in our table is our problem, not damage to the file.
        let mut bytes = good_gguf();
        // Overwrite the tensor's type id with one ggml does not define.
        let marker = 12u32.to_le_bytes();
        let at = bytes.windows(4).position(|w| w == marker).unwrap();
        bytes[at..at + 4].copy_from_slice(&250u32.to_le_bytes());
        let path = write("unknowntype.gguf", &bytes);
        let report = verify(&path).unwrap();
        assert!(report.is_intact(), "findings: {:?}", report.findings);
        assert!(report.findings.iter().any(|f| f.level == Level::Warning));
        let _ = std::fs::remove_file(&path);
    }

    // -- safetensors --------------------------------------------------------

    fn safetensors_file(header: &str, data: usize) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + data, 0);
        out
    }

    #[test]
    fn a_well_formed_safetensors_reads_clean() {
        let header = r#"{"a":{"dtype":"F16","shape":[4,8],"data_offsets":[0,64]}}"#;
        let path = write("good.safetensors", &safetensors_file(header, 64));
        let report = verify(&path).unwrap();
        assert!(report.is_intact(), "findings: {:?}", report.findings);
        assert_eq!(report.format, Format::Safetensors);
        assert_eq!(report.tensor_count, 1);
        assert_eq!(report.parameters, 32);
        assert_eq!(report.tensor_types[0].name, "F16");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_shape_that_disagrees_with_its_byte_span_is_an_error() {
        // 4x8 F16 is 64 bytes; the header claims 32.
        let header = r#"{"a":{"dtype":"F16","shape":[4,8],"data_offsets":[0,32]}}"#;
        let path = write("mismatch.safetensors", &safetensors_file(header, 32));
        let report = verify(&path).unwrap();
        assert!(!report.is_intact());
        assert!(
            report.findings[0].message.contains("needs 64 bytes"),
            "{:?}",
            report.findings
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn overlapping_tensors_are_caught_even_though_each_one_fits() {
        let header = concat!(
            r#"{"a":{"dtype":"U8","shape":[64],"data_offsets":[0,64]},"#,
            r#""b":{"dtype":"U8","shape":[64],"data_offsets":[32,96]}}"#
        );
        let path = write("overlap.safetensors", &safetensors_file(header, 96));
        let report = verify(&path).unwrap();
        assert!(!report.is_intact());
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.message.contains("overlap")),
            "{:?}",
            report.findings
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_header_larger_than_the_file_is_refused() {
        let mut bytes = (u64::MAX).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        let path = write("bigheader.safetensors", &bytes);
        let report = verify(&path).unwrap();
        assert!(!report.is_intact());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_file_that_is_neither_format_is_reported_as_such() {
        let path = write("random.bin", b"this is not a model file at all");
        let error = verify(&path).unwrap_err();
        assert!(error.contains("not a GGUF or safetensors file"), "{error}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_file_is_an_error_not_a_panic() {
        assert!(verify(&temp("absent.gguf")).is_err());
    }

    #[test]
    fn sizes_are_rendered_in_the_unit_a_person_would_use() {
        assert_eq!(human(512), "512 B");
        assert_eq!(human(1536), "1.5 KB");
        assert_eq!(human(4 * 1024 * 1024 * 1024), "4.0 GB");
    }
}
