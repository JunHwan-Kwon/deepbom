use std::cell::RefCell;

pub(super) struct Fb<'a> {
    pub(super) data: &'a [u8],
    _verification: fb_verification::Token,
    referenced_ranges: RefCell<Vec<BufferDataLocation>>,
}

mod fb_verification {
    use super::{Fb, RefCell};

    pub(super) struct Token {
        _private: (),
    }

    fn unchecked(data: &[u8]) -> Fb<'_> {
        Fb {
            data,
            _verification: Token { _private: () },
            referenced_ranges: RefCell::new(Vec::new()),
        }
    }

    pub(super) fn verified_tflite(data: &[u8]) -> Result<Fb<'_>, String> {
        let fb = unchecked(data);
        fb.record_range(0, data.len().min(8));
        fb.require_tflite()?;
        let model = fb.root_table()?;
        fb.require_bounded_graph(model)?;
        Ok(fb)
    }

    pub(super) fn verified_flatbuffer_root(data: &[u8]) -> Result<Fb<'_>, String> {
        let fb = unchecked(data);
        let root = fb.root_table()?;
        if !fb.table_within_file(root) {
            return Err("FlatBuffer root table is truncated or corrupt".to_string());
        }
        Ok(fb)
    }

    #[cfg(test)]
    pub(super) fn unchecked_for_test(data: &[u8]) -> Fb<'_> {
        unchecked(data)
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct BufferDataLocation {
    pub(super) offset: usize,
    pub(super) length: usize,
}

#[derive(Default)]
pub(super) struct ParsedConversionMetadata {
    pub(super) status: String,
    pub(super) tensorflow_version: String,
    pub(super) api_version: Option<u32>,
    pub(super) model_type: String,
    pub(super) optimization_mode_codes: Vec<i32>,
    pub(super) optimization_modes: Vec<String>,
}

pub(super) fn parse_conversion_metadata(data: &[u8]) -> ParsedConversionMetadata {
    let parsed = (|| -> Result<ParsedConversionMetadata, String> {
        let fb = Fb::verified_flatbuffer_root(data)?;
        let root = fb.root_table()?;
        let environment = fb
            .checked_table_field(root, 0, "ConversionMetadata.environment")?
            .ok_or_else(|| "environment table is missing".to_string())?;
        let tensorflow_version = fb
            .checked_string_field(environment, 0, "Environment.tensorflow_version")?
            .unwrap_or_default();
        let api_version = fb.checked_u32_field(environment, 1, 0, "Environment.api_version")?;
        let model_type_code = fb.checked_i32_field(environment, 2, 0, "Environment.model_type")?;
        let model_type = match model_type_code {
            0 => "NONE".to_string(),
            1 => "TF_SAVED_MODEL".to_string(),
            2 => "KERAS_MODEL".to_string(),
            3 => "TF_CONCRETE_FUNCTIONS".to_string(),
            4 => "TF_GRAPH_DEF".to_string(),
            5 => "TF_SESSION".to_string(),
            6 => "JAX".to_string(),
            7 => "PYTORCH".to_string(),
            other => format!("UNKNOWN({other})"),
        };
        let optimization_mode_codes =
            match fb.checked_table_field(root, 1, "ConversionMetadata.options")? {
                Some(options) => {
                    fb.checked_vector_i32(options, 0, "ConversionOptions.model_optimization_modes")?
                }
                None => Vec::new(),
            };
        let optimization_modes = optimization_mode_codes
            .iter()
            .map(|value| conversion_optimization_mode_name(*value))
            .collect();
        Ok(ParsedConversionMetadata {
            status: "parsed".to_string(),
            tensorflow_version,
            api_version: Some(api_version),
            model_type,
            optimization_mode_codes,
            optimization_modes,
        })
    })();
    match parsed {
        Ok(value) => value,
        Err(error) => ParsedConversionMetadata {
            status: format!("invalid_flatbuffer: {error}"),
            ..ParsedConversionMetadata::default()
        },
    }
}

pub(super) fn conversion_optimization_mode_name(value: i32) -> String {
    match value {
        1001 => "PTQ_FLOAT16".to_string(),
        1002 => "PTQ_DYNAMIC_RANGE".to_string(),
        1003 => "PTQ_FULL_INTEGER".to_string(),
        1004 => "PTQ_INT16".to_string(),
        2000 => "QUANTIZATION_AWARE_TRAINING".to_string(),
        3001 => "RANDOM_SPARSITY".to_string(),
        3002 => "BLOCK_SPARSITY".to_string(),
        3003 => "STRUCTURED_SPARSITY".to_string(),
        other => format!("UNKNOWN({other})"),
    }
}

impl<'a> Fb<'a> {
    pub(super) fn verified_tflite(data: &'a [u8]) -> Result<Self, String> {
        fb_verification::verified_tflite(data)
    }

    pub(super) fn verified_flatbuffer_root(data: &'a [u8]) -> Result<Self, String> {
        fb_verification::verified_flatbuffer_root(data)
    }

    #[cfg(test)]
    pub(super) fn new_for_test(data: &'a [u8]) -> Self {
        fb_verification::unchecked_for_test(data)
    }

    pub(super) fn require_tflite(&self) -> Result<(), String> {
        if self.data.len() < 8 {
            return Err("File is too small to be a TFLite FlatBuffer".to_string());
        }
        // Reject obvious non-TFLite containers (ZIP/task bundles start with "PK",
        // PNG with "\x89PNG", ONNX protobuf with 0x08, etc.)
        let magic4 = &self.data[0..4];
        if magic4 == b"PK\x03\x04" || magic4 == b"PK\x05\x06" {
            return Err("File appears to be a ZIP/task bundle, not a raw TFLite FlatBuffer. Extract the .tflite from inside the .task archive.".to_string());
        }
        // Accept TFL3 (standard TFLite v3 identifier) or no identifier
        // (older TFLite converters / some frameworks omit the 4-byte identifier).
        let ident = &self.data[4..8];
        if ident != b"TFL3" && ident != b"TFL2" {
            // Verify the root offset looks plausible for an identifierless FlatBuffer.
            let root = u32::from_le_bytes([self.data[0], self.data[1], self.data[2], self.data[3]])
                as usize;
            if root == 0 || root >= self.data.len() {
                return Err(format!(
                    "Not a recognized TFLite FlatBuffer (identifier: {:?}; root offset {})",
                    std::str::from_utf8(ident).unwrap_or("?"),
                    root
                ));
            }
        }
        Ok(())
    }

    pub(super) fn root_table(&self) -> Result<usize, String> {
        self.record_range(0, 4);
        let root = self.u32(0).ok_or("Missing FlatBuffer root table")? as usize;
        if root >= self.data.len() {
            return Err("FlatBuffer root table points outside the file".to_string());
        }
        Ok(root)
    }

    pub(super) fn u8(&self, pos: usize) -> Option<u8> {
        self.data.get(pos).copied()
    }

    pub(super) fn i8(&self, pos: usize) -> Option<i8> {
        self.u8(pos).map(|value| value as i8)
    }

    pub(super) fn u16(&self, pos: usize) -> Option<u16> {
        let bytes = self.data.get(pos..pos + 2)?;
        Some(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(super) fn i32(&self, pos: usize) -> Option<i32> {
        let bytes = self.data.get(pos..pos + 4)?;
        Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(super) fn u32(&self, pos: usize) -> Option<u32> {
        let bytes = self.data.get(pos..pos + 4)?;
        Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(super) fn i64(&self, pos: usize) -> Option<i64> {
        let bytes = self.data.get(pos..pos + 8)?;
        Some(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    pub(super) fn u64(&self, pos: usize) -> Option<u64> {
        let bytes = self.data.get(pos..pos + 8)?;
        Some(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    pub(super) fn f32(&self, pos: usize) -> Option<f32> {
        let bytes = self.data.get(pos..pos + 4)?;
        Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(super) fn vtable(&self, table: usize) -> Option<usize> {
        let offset = self.i32(table)? as isize;
        let pos = table as isize - offset;
        if pos < 0 {
            return None;
        }
        Some(pos as usize)
    }

    pub(super) fn field_pos(&self, table: usize, field_index: usize) -> Option<usize> {
        let vtable = self.vtable(table)?;
        let vtable_offset = 4 + field_index * 2;
        let vtable_size = self.u16(vtable)? as usize;
        if vtable_offset >= vtable_size {
            return None;
        }
        let offset = self.u16(vtable + vtable_offset)? as usize;
        if offset == 0 {
            None
        } else {
            Some(table + offset)
        }
    }

    pub(super) fn table_field(&self, table: usize, field_index: usize) -> Option<usize> {
        let pos = self.field_pos(table, field_index)?;
        Some(pos + self.u32(pos)? as usize)
    }

    pub(super) fn string_field(&self, table: usize, field_index: usize) -> Option<String> {
        let pos = self.field_pos(table, field_index)?;
        let string_pos = pos + self.u32(pos)? as usize;
        let len = self.u32(string_pos)? as usize;
        let bytes = self.data.get(string_pos + 4..string_pos + 4 + len)?;
        self.record_range(string_pos, 4usize.checked_add(len)?.checked_add(1)?);
        Some(String::from_utf8_lossy(bytes).to_string())
    }

    pub(super) fn checked_field_pos(
        &self,
        table: usize,
        field_index: usize,
        width: usize,
        label: &str,
    ) -> Result<Option<usize>, String> {
        if !self.table_within_file(table) {
            return Err(format!("{label} table is truncated or corrupt"));
        }
        let Some(position) = self.field_pos(table, field_index) else {
            return Ok(None);
        };
        let vtable = self
            .vtable(table)
            .ok_or_else(|| format!("{label} vtable is invalid"))?;
        let table_size =
            self.u16(vtable + 2)
                .ok_or_else(|| format!("{label} table size is truncated"))? as usize;
        let table_end = table
            .checked_add(table_size)
            .ok_or_else(|| format!("{label} table size overflows"))?;
        let field_end = position
            .checked_add(width)
            .ok_or_else(|| format!("{label} field offset overflows"))?;
        if position < table || field_end > table_end || field_end > self.data.len() {
            return Err(format!("{label} field is truncated or outside its table"));
        }
        Ok(Some(position))
    }

    pub(super) fn checked_table_field(
        &self,
        table: usize,
        field_index: usize,
        label: &str,
    ) -> Result<Option<usize>, String> {
        let Some(position) = self.checked_field_pos(table, field_index, 4, label)? else {
            return Ok(None);
        };
        let target = position
            .checked_add(
                self.u32(position)
                    .ok_or_else(|| format!("{label} table offset is truncated"))?
                    as usize,
            )
            .ok_or_else(|| format!("{label} table offset overflows"))?;
        if !self.table_within_file(target) {
            return Err(format!("{label} referenced table is truncated or corrupt"));
        }
        Ok(Some(target))
    }

    pub(super) fn checked_string_field(
        &self,
        table: usize,
        field_index: usize,
        label: &str,
    ) -> Result<Option<String>, String> {
        let Some(position) = self.checked_field_pos(table, field_index, 4, label)? else {
            return Ok(None);
        };
        let string_position = position
            .checked_add(
                self.u32(position)
                    .ok_or_else(|| format!("{label} string offset is truncated"))?
                    as usize,
            )
            .ok_or_else(|| format!("{label} string offset overflows"))?;
        let length =
            self.u32(string_position)
                .ok_or_else(|| format!("{label} string length is truncated"))? as usize;
        let start = string_position
            .checked_add(4)
            .ok_or_else(|| format!("{label} string start overflows"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| format!("{label} string length overflows"))?;
        let bytes = self
            .data
            .get(start..end)
            .ok_or_else(|| format!("{label} string extends past the end of the buffer"))?;
        if self.data.get(end).copied() != Some(0) {
            return Err(format!("{label} string terminator is missing"));
        }
        self.record_range(string_position, end + 1 - string_position);
        let value =
            std::str::from_utf8(bytes).map_err(|_| format!("{label} string is not valid UTF-8"))?;
        Ok(Some(value.to_string()))
    }

    pub(super) fn checked_u32_field(
        &self,
        table: usize,
        field_index: usize,
        default: u32,
        label: &str,
    ) -> Result<u32, String> {
        let Some(position) = self.checked_field_pos(table, field_index, 4, label)? else {
            return Ok(default);
        };
        self.u32(position)
            .ok_or_else(|| format!("{label} scalar is truncated"))
    }

    pub(super) fn checked_i8_field(
        &self,
        table: usize,
        field_index: usize,
        default: i8,
        label: &str,
    ) -> Result<i8, String> {
        let Some(position) = self.checked_field_pos(table, field_index, 1, label)? else {
            return Ok(default);
        };
        self.i8(position)
            .ok_or_else(|| format!("{label} scalar is truncated"))
    }

    pub(super) fn checked_i32_field(
        &self,
        table: usize,
        field_index: usize,
        default: i32,
        label: &str,
    ) -> Result<i32, String> {
        let Some(position) = self.checked_field_pos(table, field_index, 4, label)? else {
            return Ok(default);
        };
        self.i32(position)
            .ok_or_else(|| format!("{label} scalar is truncated"))
    }

    pub(super) fn checked_u64_field(
        &self,
        table: usize,
        field_index: usize,
        default: u64,
        label: &str,
    ) -> Result<u64, String> {
        let Some(position) = self.checked_field_pos(table, field_index, 8, label)? else {
            return Ok(default);
        };
        self.u64(position)
            .ok_or_else(|| format!("{label} scalar is truncated"))
    }

    pub(super) fn checked_vector_i32(
        &self,
        table: usize,
        field_index: usize,
        label: &str,
    ) -> Result<Vec<i32>, String> {
        let Some(position) = self.checked_field_pos(table, field_index, 4, label)? else {
            return Ok(Vec::new());
        };
        let vector_position = position
            .checked_add(
                self.u32(position)
                    .ok_or_else(|| format!("{label} vector offset is truncated"))?
                    as usize,
            )
            .ok_or_else(|| format!("{label} vector offset overflows"))?;
        let length =
            self.u32(vector_position)
                .ok_or_else(|| format!("{label} vector length is truncated"))? as usize;
        let start = vector_position
            .checked_add(4)
            .ok_or_else(|| format!("{label} vector start overflows"))?;
        let end = start
            .checked_add(
                length
                    .checked_mul(4)
                    .ok_or_else(|| format!("{label} vector length overflows"))?,
            )
            .ok_or_else(|| format!("{label} vector extent overflows"))?;
        if end > self.data.len() {
            return Err(format!("{label} vector extends past the end of the buffer"));
        }
        self.record_range(vector_position, end - vector_position);
        (0..length)
            .map(|index| {
                self.i32(start + index * 4)
                    .ok_or_else(|| format!("{label} vector element is truncated"))
            })
            .collect()
    }

    /// The vector's declared extent, without checking it against the file. A
    /// declared extent that does not fit is exactly what truncation looks like,
    /// so the integrity gate needs to observe it before it is filtered away.
    pub(super) fn vector_extent(
        &self,
        table: usize,
        field_index: usize,
        stride: usize,
    ) -> Option<(usize, usize, usize)> {
        let pos = self.field_pos(table, field_index)?;
        let vector_pos = pos.checked_add(self.u32(pos)? as usize)?;
        let len = self.u32(vector_pos)? as usize;
        let start = vector_pos.checked_add(4)?;
        let end = start.checked_add(len.checked_mul(stride)?)?;
        Some((start, len, end))
    }

    /// Declared extents that run past the end of the file are rejected outright
    /// rather than read element-by-element, which would silently drop the
    /// unreadable tail and make a truncated artifact look like a smaller model.
    pub(super) fn vector_location(
        &self,
        table: usize,
        field_index: usize,
        stride: usize,
    ) -> Option<(usize, usize)> {
        let (start, len, end) = self.vector_extent(table, field_index, stride)?;
        if end > self.data.len() {
            return None;
        }
        self.record_range(start.saturating_sub(4), end - start.saturating_sub(4));
        Some((start, len))
    }

    /// A FlatBuffer table is only readable if its vtable and its own declared
    /// inline size both fit in the file. Scalar field reads default to 0 when
    /// they fall off the end (`unwrap_or(0)` throughout this module), which
    /// turns a truncated artifact into a plausible but wrong one — an operator
    /// code that reads as 0 silently reclassifies the op and zeroes its MACs.
    pub(super) fn table_within_file(&self, table: usize) -> bool {
        let Some(vtable) = self.vtable(table) else {
            return false;
        };
        let Some(vtable_size) = self.u16(vtable).map(usize::from) else {
            return false;
        };
        let Some(table_size) = self.u16(vtable + 2).map(usize::from) else {
            return false;
        };
        if vtable_size < 4 || vtable_size % 2 != 0 || table_size < 4 {
            return false;
        }
        let Some(vtable_end) = vtable.checked_add(vtable_size) else {
            return false;
        };
        let Some(table_end) = table.checked_add(table_size) else {
            return false;
        };
        if vtable_end > self.data.len() || table_end > self.data.len() {
            return false;
        }
        let valid = (4..vtable_size).step_by(2).all(|entry| {
            self.u16(vtable + entry).is_some_and(|offset| {
                offset == 0 || (usize::from(offset) >= 4 && usize::from(offset) < table_size)
            })
        });
        if valid {
            self.record_range(vtable, vtable_size);
            self.record_range(table, table_size);
        }
        valid
    }

    pub(super) fn vector_overflows(&self, table: usize, field_index: usize, stride: usize) -> bool {
        match self.vector_extent(table, field_index, stride) {
            Some((start, _, end)) => {
                if end <= self.data.len() {
                    self.record_range(start.saturating_sub(4), end - start.saturating_sub(4));
                    false
                } else {
                    true
                }
            }
            // A field that cannot be located at all is absent, not truncated;
            // the surrounding parse already treats that as "not present".
            None => self.field_pos(table, field_index).is_some(),
        }
    }

    /// Fail closed when the FlatBuffer declares graph vectors the file cannot
    /// contain. Without this a truncated artifact still parses, and every
    /// derived total (MACs, quantization class, delegation) is silently wrong.
    pub(super) fn require_bounded_graph(&self, model: usize) -> Result<(), String> {
        let truncated = |label: &str| {
            Err(format!(
                "TFLite {label} extends past the end of the file; the artifact is truncated or corrupt"
            ))
        };
        if !self.table_within_file(model) {
            return truncated("model table");
        }
        self.checked_u32_field(model, 0, 0, "Model.version")?;
        self.checked_string_field(model, 3, "Model.description")?;
        for (field, label) in [
            (1usize, "operator_codes"),
            (2usize, "subgraphs"),
            (4usize, "buffers"),
            (6usize, "metadata"),
            (7usize, "signature_defs"),
            (8usize, "external_buffer_groups"),
            (9usize, "external_buffers"),
        ] {
            if self.vector_overflows(model, field, 4) {
                return Err(format!(
                    "TFLite {label} vector extends past the end of the file; the artifact is truncated or corrupt"
                ));
            }
        }
        for operator_code in self.vector_tables(model, 1) {
            if !self.table_within_file(operator_code) {
                return truncated("operator_code table");
            }
            self.checked_i8_field(operator_code, 0, 0, "OperatorCode.deprecated_builtin_code")?;
            self.checked_string_field(operator_code, 1, "OperatorCode.custom_code")?;
            self.checked_i32_field(operator_code, 2, 1, "OperatorCode.version")?;
            self.checked_i32_field(operator_code, 3, 0, "OperatorCode.builtin_code")?;
        }
        for subgraph in self.vector_tables(model, 2) {
            if !self.table_within_file(subgraph) {
                return truncated("subgraph table");
            }
            self.checked_string_field(subgraph, 4, "SubGraph.name")?;
            for (field, label) in [
                (0usize, "tensors"),
                (1usize, "subgraph inputs"),
                (2usize, "subgraph outputs"),
                (3usize, "operators"),
            ] {
                if self.vector_overflows(subgraph, field, 4) {
                    return Err(format!(
                        "TFLite {label} vector extends past the end of the file; the artifact is truncated or corrupt"
                    ));
                }
            }
            // Per-tensor shape and quantization vectors are what every derived
            // total is computed from. A lost shape silently zeroes that tensor's
            // MAC contribution, so their extents are checked individually.
            for tensor in self.vector_tables(subgraph, 0) {
                if !self.table_within_file(tensor) {
                    return truncated("tensor table");
                }
                let tensor_type = self.checked_i8_field(tensor, 1, 0, "Tensor.type")?;
                self.checked_u32_field(tensor, 2, 0, "Tensor.buffer")?;
                self.checked_string_field(tensor, 3, "Tensor.name")?;
                self.checked_i8_field(tensor, 5, 0, "Tensor.is_variable")?;
                self.checked_i8_field(tensor, 8, 0, "Tensor.has_rank")?;
                let external_buffer =
                    self.checked_u32_field(tensor, 10, 0, "Tensor.external_buffer")?;
                if external_buffer != 0 {
                    return Err(format!(
                        "TFLite Tensor.external_buffer references external buffer {external_buffer}; external TFLite weight sidecars are not supported by this single-artifact analyzer"
                    ));
                }
                if tensor_type == 14 {
                    return Err(
                        "TFLite VARIANT tensor subtypes are not supported by this analyzer"
                            .to_string(),
                    );
                }
                for (field, stride, label) in [
                    (0usize, 4usize, "tensor shape"),
                    (7, 4, "tensor shape_signature"),
                    (9, 4, "tensor variant_tensors"),
                ] {
                    if self.vector_overflows(tensor, field, stride) {
                        return Err(format!(
                            "TFLite {label} vector extends past the end of the file; the artifact is truncated or corrupt"
                        ));
                    }
                }
                if let Some(sparsity) = self.checked_table_field(tensor, 6, "Tensor.sparsity")? {
                    self.require_bounded_sparsity(sparsity)?;
                }
                if let Some(quant) = self.checked_table_field(tensor, 4, "Tensor.quantization")? {
                    if !self.table_within_file(quant) {
                        return truncated("tensor quantization table");
                    }
                    let details_type =
                        self.checked_i8_field(quant, 4, 0, "QuantizationParameters.details_type")?;
                    let details =
                        self.checked_table_field(quant, 5, "QuantizationParameters.details")?;
                    if (details_type == 0) != details.is_none() {
                        return Err(
                            "TFLite QuantizationParameters details union type/table mismatch"
                                .to_string(),
                        );
                    }
                    if details_type != 0 {
                        return Err(format!(
                            "TFLite quantization details type {details_type} is not supported by the affine constant decoder"
                        ));
                    }
                    self.checked_i32_field(
                        quant,
                        6,
                        0,
                        "QuantizationParameters.quantized_dimension",
                    )?;
                    for (field, stride, label) in [
                        (0usize, 4usize, "quantization min"),
                        (1, 4, "quantization max"),
                        (2, 4, "quantization scale"),
                        (3, 8, "quantization zero_point"),
                    ] {
                        if self.vector_overflows(quant, field, stride) {
                            return Err(format!(
                                "TFLite {label} vector extends past the end of the file; the artifact is truncated or corrupt"
                            ));
                        }
                    }
                }
            }
            for operator in self.vector_tables(subgraph, 3) {
                if !self.table_within_file(operator) {
                    return truncated("operator table");
                }
                let opcode_index =
                    self.checked_u32_field(operator, 0, 0, "Operator.opcode_index")? as usize;
                self.checked_i8_field(operator, 6, 0, "Operator.custom_options_format")?;
                let large_custom_offset =
                    self.checked_u64_field(operator, 9, 0, "Operator.large_custom_options_offset")?;
                let large_custom_size =
                    self.checked_u64_field(operator, 10, 0, "Operator.large_custom_options_size")?;
                self.checked_i32_field(operator, 13, -1, "Operator.debug_metadata_index")?;
                self.require_bounded_external_slice(
                    large_custom_offset,
                    large_custom_size,
                    "Operator.large_custom_options",
                )?;
                for (field, stride, label) in [
                    (1usize, 4usize, "operator inputs"),
                    (2usize, 4usize, "operator outputs"),
                    (5usize, 1usize, "operator custom_options"),
                    (7usize, 1usize, "operator mutating_variable_inputs"),
                    (8usize, 4usize, "operator intermediates"),
                ] {
                    if self.vector_overflows(operator, field, stride) {
                        return Err(format!(
                            "TFLite {label} vector extends past the end of the file; the artifact is truncated or corrupt"
                        ));
                    }
                }
                let options_type =
                    self.checked_i8_field(operator, 3, 0, "Operator.builtin_options_type")?;
                let options = self.checked_table_field(operator, 4, "Operator.builtin_options")?;
                if (options_type == 0) != options.is_none() {
                    return Err(
                        "TFLite Operator builtin_options union type/table mismatch".to_string()
                    );
                }
                if let Some(options) = options {
                    self.require_bounded_builtin_options(options_type, options)?;
                }
                let options_2_type =
                    self.checked_i8_field(operator, 11, 0, "Operator.builtin_options_2_type")?;
                let options_2 =
                    self.checked_table_field(operator, 12, "Operator.builtin_options_2")?;
                if (options_2_type == 0) != options_2.is_none() {
                    return Err(
                        "TFLite Operator builtin_options_2 union type/table mismatch".to_string(),
                    );
                }
                if let Some(options_2) = options_2 {
                    if !self.table_within_file(options_2) {
                        return truncated("operator builtin_options_2 table");
                    }
                }
                if let Some(expected) = self.expected_builtin_options_type(opcode_index, model)? {
                    if options_type != 0 && options_type != expected {
                        return Err(format!(
                            "TFLite operator opcode index {opcode_index} requires builtin_options type {expected}, found {options_type}"
                        ));
                    }
                }
            }
        }
        for buffer in self.vector_tables(model, 4) {
            if !self.table_within_file(buffer) {
                return truncated("buffer table");
            }
            if self.vector_overflows(buffer, 0, 1) {
                return truncated("constant buffer");
            }
            let offset = self.checked_u64_field(buffer, 1, 0, "Buffer.offset")?;
            let size = self.checked_u64_field(buffer, 2, 0, "Buffer.size")?;
            self.require_bounded_external_slice(offset, size, "Buffer external data")?;
        }
        for metadata in self.vector_tables(model, 6) {
            if !self.table_within_file(metadata) {
                return truncated("metadata table");
            }
            self.checked_string_field(metadata, 0, "Metadata.name")?;
            self.checked_u32_field(metadata, 1, 0, "Metadata.buffer")?;
        }
        for signature in self.vector_tables(model, 7) {
            if !self.table_within_file(signature) {
                return truncated("signature_def table");
            }
            for (field, label) in [(0usize, "signature inputs"), (1usize, "signature outputs")] {
                if self.vector_overflows(signature, field, 4) {
                    return Err(format!(
                        "TFLite {label} vector extends past the end of the file; the artifact is truncated or corrupt"
                    ));
                }
                for tensor_map in self.vector_tables(signature, field) {
                    if !self.table_within_file(tensor_map) {
                        return truncated("signature tensor_map table");
                    }
                    self.checked_string_field(tensor_map, 0, "TensorMap.name")?;
                    self.checked_u32_field(tensor_map, 1, 0, "TensorMap.tensor_index")?;
                }
            }
            self.checked_string_field(signature, 2, "SignatureDef.signature_key")?;
            self.checked_u32_field(signature, 3, 0, "SignatureDef.subgraph_index")?;
        }
        for group in self.vector_tables(model, 8) {
            if !self.table_within_file(group) {
                return truncated("external_buffer_group table");
            }
            self.checked_string_field(group, 0, "ExternalBufferGroup.name")?;
        }
        for external in self.vector_tables(model, 9) {
            if !self.table_within_file(external) {
                return truncated("external_buffer table");
            }
            self.checked_u32_field(external, 0, 0, "ExternalBuffer.id")?;
            self.checked_u32_field(external, 1, 0, "ExternalBuffer.group")?;
            self.checked_u64_field(external, 2, 0, "ExternalBuffer.offset")?;
            self.checked_u64_field(external, 3, 0, "ExternalBuffer.length")?;
            self.checked_string_field(external, 4, "ExternalBuffer.packing")?;
        }
        Ok(())
    }

    fn require_bounded_sparsity(&self, table: usize) -> Result<(), String> {
        if !self.table_within_file(table) {
            return Err("TFLite SparsityParameters table is truncated or corrupt".to_string());
        }
        for (field, label) in [
            (0usize, "SparsityParameters.traversal_order"),
            (1usize, "SparsityParameters.block_map"),
        ] {
            if self.vector_overflows(table, field, 4) {
                return Err(format!("TFLite {label} vector extends past the artifact"));
            }
        }
        if self.vector_overflows(table, 2, 4) {
            return Err(
                "TFLite SparsityParameters.dim_metadata vector extends past the artifact"
                    .to_string(),
            );
        }
        let traversal_count = self
            .vector_location(table, 0, 4)
            .map(|(_, length)| length)
            .unwrap_or(0);
        let dimensions = self.vector_tables(table, 2);
        if traversal_count != dimensions.len() {
            return Err(format!(
                "TFLite sparse traversal/dimension cardinality mismatch: {traversal_count} traversal entries and {} dimension metadata records",
                dimensions.len()
            ));
        }
        for dimension in dimensions {
            if !self.table_within_file(dimension) {
                return Err("TFLite DimensionMetadata table is truncated or corrupt".to_string());
            }
            let format = self.checked_i8_field(dimension, 0, 0, "DimensionMetadata.format")?;
            if !matches!(format, 0 | 1) {
                return Err(format!(
                    "TFLite DimensionMetadata.format {format} is unsupported"
                ));
            }
            self.checked_i32_field(dimension, 1, 0, "DimensionMetadata.dense_size")?;
            let segments_type =
                self.checked_i8_field(dimension, 2, 0, "DimensionMetadata.array_segments_type")?;
            let segments =
                self.checked_table_field(dimension, 3, "DimensionMetadata.array_segments")?;
            let indices_type =
                self.checked_i8_field(dimension, 4, 0, "DimensionMetadata.array_indices_type")?;
            let indices =
                self.checked_table_field(dimension, 5, "DimensionMetadata.array_indices")?;
            self.require_bounded_sparse_index_vector(
                segments_type,
                segments,
                "DimensionMetadata.array_segments",
            )?;
            self.require_bounded_sparse_index_vector(
                indices_type,
                indices,
                "DimensionMetadata.array_indices",
            )?;
            if format == 0 && (segments_type != 0 || indices_type != 0) {
                return Err(
                    "TFLite dense DimensionMetadata unexpectedly declares sparse index vectors"
                        .to_string(),
                );
            }
            if format == 1 && (segments_type == 0 || indices_type == 0) {
                return Err(
                    "TFLite SPARSE_CSR DimensionMetadata is missing index vectors".to_string(),
                );
            }
        }
        Ok(())
    }

    fn require_bounded_sparse_index_vector(
        &self,
        vector_type: i8,
        table: Option<usize>,
        label: &str,
    ) -> Result<(), String> {
        if (vector_type == 0) != table.is_none() {
            return Err(format!("TFLite {label} union type/table mismatch"));
        }
        let Some(table) = table else {
            return Ok(());
        };
        let stride = match vector_type {
            1 => 4,
            2 => 2,
            3 => 1,
            other => {
                return Err(format!(
                    "TFLite {label} sparse index vector type {other} is unsupported"
                ))
            }
        };
        if self.vector_overflows(table, 0, stride) {
            return Err(format!("TFLite {label} values extend past the artifact"));
        }
        Ok(())
    }

    fn require_bounded_external_slice(
        &self,
        offset: u64,
        size: u64,
        label: &str,
    ) -> Result<(), String> {
        if offset == 0 && size == 0 {
            return Ok(());
        }
        if offset <= 1 || size == 0 {
            return Err(format!("{label} offset/size pair is inconsistent"));
        }
        let offset = usize::try_from(offset)
            .map_err(|_| format!("{label} offset does not fit the host address space"))?;
        let size = usize::try_from(size)
            .map_err(|_| format!("{label} size does not fit the host address space"))?;
        let end = offset
            .checked_add(size)
            .ok_or_else(|| format!("{label} extent overflows"))?;
        if end > self.data.len() {
            return Err(format!("{label} extends past the end of the artifact"));
        }
        self.record_range(offset, size);
        Ok(())
    }

    fn require_bounded_builtin_options(
        &self,
        options_type: i8,
        table: usize,
    ) -> Result<(), String> {
        // Widths follow the pinned TensorFlow schema. Only option records whose
        // scalars feed analysis or delegation decisions need field-level checks;
        // every other option table is still structurally bounded above.
        let fields: &[(usize, usize)] = match options_type {
            1 => &[(0, 1), (1, 4), (2, 4), (3, 1), (4, 4), (5, 4), (6, 1)],
            2 => &[(0, 1), (1, 4), (2, 4), (3, 4), (4, 1), (5, 4), (6, 4)],
            5 => &[(0, 1), (1, 4), (2, 4), (3, 4), (4, 4), (5, 1)],
            8 => &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
            9 | 75 => &[(0, 4)],
            10 => &[(0, 4), (1, 1)],
            11 | 12 | 21 | 28 | 29 => &[(0, 1)],
            19 | 94 => &[(0, 4)],
            32 => &[(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 1)],
            _ => &[],
        };
        for (field, width) in fields {
            self.checked_field_pos(
                table,
                *field,
                *width,
                &format!("BuiltinOptions({options_type}).field[{field}]"),
            )?;
        }
        Ok(())
    }

    fn expected_builtin_options_type(
        &self,
        opcode_index: usize,
        model: usize,
    ) -> Result<Option<i8>, String> {
        let operator_codes = self.vector_tables(model, 1);
        let Some(operator_code) = operator_codes.get(opcode_index).copied() else {
            return Err(format!(
                "TFLite Operator.opcode_index {opcode_index} is outside Model.operator_codes"
            ));
        };
        let deprecated =
            self.checked_i8_field(operator_code, 0, 0, "OperatorCode.deprecated_builtin_code")?
                as i32;
        let builtin = self.checked_i32_field(operator_code, 3, 0, "OperatorCode.builtin_code")?;
        let code = if builtin < 127 { deprecated } else { builtin };
        Ok(match code {
            0 => Some(11),
            1 | 12 | 17 => Some(5),
            2 => Some(10),
            3 => Some(1),
            4 => Some(2),
            5 => Some(94),
            9 => Some(8),
            11 => Some(12),
            18 => Some(21),
            25 => Some(9),
            26 => Some(19),
            41 => Some(28),
            42 => Some(29),
            45 => Some(32),
            98 => Some(75),
            _ => None,
        })
    }

    pub(super) fn vector_i32(&self, table: usize, field_index: usize) -> Vec<i32> {
        let Some((start, len)) = self.vector_location(table, field_index, 4) else {
            return Vec::new();
        };
        (0..len).filter_map(|i| self.i32(start + i * 4)).collect()
    }

    pub(super) fn vector_i64(&self, table: usize, field_index: usize) -> Vec<i64> {
        let Some((start, len)) = self.vector_location(table, field_index, 8) else {
            return Vec::new();
        };
        (0..len).filter_map(|i| self.i64(start + i * 8)).collect()
    }

    pub(super) fn vector_f32(&self, table: usize, field_index: usize) -> Vec<f32> {
        let Some((start, len)) = self.vector_location(table, field_index, 4) else {
            return Vec::new();
        };
        (0..len).filter_map(|i| self.f32(start + i * 4)).collect()
    }

    pub(super) fn vector_tables(&self, table: usize, field_index: usize) -> Vec<usize> {
        let Some((start, len)) = self.vector_location(table, field_index, 4) else {
            return Vec::new();
        };
        (0..len)
            .filter_map(|i| {
                let entry = start + i * 4;
                Some(entry + self.u32(entry)? as usize)
            })
            .collect()
    }

    fn record_range(&self, offset: usize, length: usize) {
        if length == 0
            || offset
                .checked_add(length)
                .is_none_or(|end| end > self.data.len())
        {
            return;
        }
        self.referenced_ranges
            .borrow_mut()
            .push(BufferDataLocation { offset, length });
    }

    pub(super) fn referenced_byte_ranges(&self) -> Vec<BufferDataLocation> {
        let mut ranges = self.referenced_ranges.borrow().clone();
        ranges.sort_by_key(|range| (range.offset, range.length));
        let mut merged = Vec::<BufferDataLocation>::new();
        for range in ranges {
            let Some(end) = range.offset.checked_add(range.length) else {
                continue;
            };
            if let Some(last) = merged.last_mut() {
                let last_end = last.offset + last.length;
                if range.offset <= last_end {
                    last.length = last.length.max(end - last.offset);
                    continue;
                }
            }
            merged.push(range);
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::Fb;

    const SAMPLE: &[u8] = include_bytes!("../web/samples/mobilenet_v2_1.0_224_quant.tflite");

    #[test]
    fn rejects_scalar_that_crosses_its_declared_table_width() {
        let mut bytes = SAMPLE.to_vec();
        let (entry, invalid_offset) = {
            let fb = Fb::new_for_test(&bytes);
            let model = fb.root_table().expect("sample root");
            let subgraph = fb.vector_tables(model, 2)[0];
            let tensor = fb.vector_tables(subgraph, 0)[0];
            let vtable = fb.vtable(tensor).expect("tensor vtable");
            let table_size = fb.u16(vtable + 2).expect("tensor table size");
            (vtable + 4 + 2 * 2, table_size - 1)
        };
        bytes[entry..entry + 2].copy_from_slice(&invalid_offset.to_le_bytes());

        let error = match Fb::verified_tflite(&bytes) {
            Ok(_) => panic!("malformed scalar was accepted"),
            Err(error) => error,
        };
        assert!(
            error.contains("Tensor.buffer field is truncated"),
            "{error}"
        );
    }

    #[test]
    fn rejects_builtin_option_scalar_that_crosses_table_width() {
        let mut bytes = SAMPLE.to_vec();
        let (entry, invalid_offset) = {
            let fb = Fb::new_for_test(&bytes);
            let model = fb.root_table().expect("sample root");
            let subgraph = fb.vector_tables(model, 2)[0];
            let operator = fb
                .vector_tables(subgraph, 3)
                .into_iter()
                .find(|table| {
                    fb.field_pos(*table, 3).and_then(|position| fb.i8(position)) == Some(1)
                })
                .expect("CONV_2D options");
            let options = fb.table_field(operator, 4).expect("options table");
            let vtable = fb.vtable(options).expect("options vtable");
            let table_size = fb.u16(vtable + 2).expect("options table size");
            (vtable + 4 + 2, table_size - 1)
        };
        bytes[entry..entry + 2].copy_from_slice(&invalid_offset.to_le_bytes());

        let error = match Fb::verified_tflite(&bytes) {
            Ok(_) => panic!("malformed builtin option scalar was accepted"),
            Err(error) => error,
        };
        assert!(
            error.contains("BuiltinOptions(1).field[1] field is truncated"),
            "{error}"
        );
    }
}
