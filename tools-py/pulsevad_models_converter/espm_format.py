"""VAD-specific ESPM v3 writer and strict reader."""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Any

MAGIC = 0x4D505345
HEADER_SIZE = 128
SECTION_ENTRY_SIZE = 40
ALIGNMENT = 64
V3_FORMAT_VERSION = 3
V3_MAX_SECTIONS = 16
V3_INDEX_HEADER_SIZE = 36
V3_TENSOR_RECORD_SIZE = 160
V3_MAX_TENSORS = 512
V3_SECTION_OPTIONAL = 1
V3_METADATA_HEADER = struct.Struct("<4sHHII")
V3_METADATA_RECORD = struct.Struct("<HBBI")
V3_INDEX_HEADER = struct.Struct("<4sHHHHIIQQ")
V3_TENSOR = struct.Struct("<II4BI8IIB3sQIQ Q64s12s")
assert V3_METADATA_HEADER.size == 16
assert V3_METADATA_RECORD.size == 8
assert V3_INDEX_HEADER.size == V3_INDEX_HEADER_SIZE
assert V3_TENSOR.size == V3_TENSOR_RECORD_SIZE
MODEL_TYPE_VAD = 1
MODEL_QUANT_FP32 = 4
SECTION_TENSOR_INDEX = 6
SECTION_TENSOR_DATA = 7
SECTION_METADATA = 8
SECTION_CONFIG_JSON = 5

_HEADER = struct.Struct("<IHHHH32s16sIIIQIHHIII28s")
_SECTION = struct.Struct("<HHQQI16s")
assert _HEADER.size == HEADER_SIZE
assert _SECTION.size == SECTION_ENTRY_SIZE

FRONTEND_CONFIG = {
    "sample_rate": 16000,
    "window_samples": 3200,
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "num_frames": 21,
    "num_mels": 64,
    "preemphasis_alpha": 0.97,
    "epsilon": 1.0e-5,
}

_STUDENT_LAYERS = {
    "adapter": ((12, 64, 1), 12),
    "conv0_dw": ((12, 1, 11), None),
    "conv0_pw": ((8, 12, 1), 8),
    "block1": ((8, 8, 1), 8),
    "block2": ((8, 8, 1), 8),
    "subA_dw": ((8, 1, 17), None),
    "subA_pw": ((8, 8, 1), 8),
    "subC_dw": ((8, 1, 17), None),
    "subC_pw": ((8, 8, 1), 8),
    "skip": ((8, 8, 1), 8),
    "conv4_dw": ((8, 1, 29), None),
    "conv4_pw": ((8, 8, 1), 8),
    "conv5": ((8, 8, 1), 8),
    "classifier": ((2, 8), 2),
}

_TEACHER_LAYERS = {
    "adapter": ((128, 64, 1), 128),
    "conv0_dw": ((128, 1, 11), None),
    "conv0_pw": ((128, 128, 1), 128),
    "block1": ((64, 128, 1), 64),
    "block2": ((64, 64, 1), 64),
    "subA_dw": ((64, 1, 17), None),
    "subA_pw": ((64, 64, 1), 64),
    "subC_dw": ((64, 1, 17), None),
    "subC_pw": ((64, 64, 1), 64),
    "skip": ((64, 64, 1), 64),
    "conv4_dw": ((64, 1, 29), None),
    "conv4_pw": ((128, 64, 1), 128),
    "conv5": ((128, 128, 1), 128),
    "classifier": ((2, 128), 2),
}

_MODEL_LAYERS = {
    "student": _STUDENT_LAYERS,
    "teacher": _TEACHER_LAYERS,
}


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def _fixed_string(value: str, capacity: int, field: str) -> bytes:
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) >= capacity or b"\0" in encoded:
        raise ValueError(f"{field} must fit in {capacity - 1} UTF-8 bytes")
    return encoded + bytes(capacity - len(encoded))


def _read_fixed_string(value: bytes, field: str) -> str:
    end = value.find(b"\0")
    if end <= 0 or any(value[end + 1:]):
        raise ValueError(f"invalid {field} string")
    try:
        return value[:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 in {field}") from error


def _float32_bytes(values: Any, field: str) -> bytes:
    if hasattr(values, "astype") and hasattr(values, "tobytes"):
        array = values.astype("<f4", copy=False)
        if not bool(array.size):
            raise ValueError(f"{field} cannot be empty")
        if not bool(__import__("numpy").isfinite(array).all()):
            raise ValueError(f"{field} contains a non-finite value")
        return array.tobytes(order="C")

    items = [float(value) for value in values]
    if not items:
        raise ValueError(f"{field} cannot be empty")
    if not all(math.isfinite(value) for value in items):
        raise ValueError(f"{field} contains a non-finite value")
    return struct.pack(f"<{len(items)}f", *items)


def _flatten(values: Any) -> Any:
    return values.reshape(-1) if hasattr(values, "reshape") else values


def _checked_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field} is out of range")
    return value








def _encode_v3_metadata(records: dict[str, tuple[int, bytes]]) -> bytes:
    encoded = bytearray(V3_METADATA_HEADER.pack(b"EMET", 1, 16, len(records), 0))
    for key in sorted(records):
        key_bytes = key.encode("utf-8")
        value_type, value = records[key]
        if not key_bytes or b"\0" in key_bytes or len(key_bytes) > 0xFFFF:
            raise ValueError(f"invalid metadata key {key!r}")
        encoded.extend(V3_METADATA_RECORD.pack(len(key_bytes), value_type, 1, len(value)))
        encoded.extend(key_bytes)
        encoded.extend(value)
    if len(encoded) > 64 * 1024:
        raise ValueError("typed metadata exceeds 64 KiB")
    return bytes(encoded)


def _meta_u32(value: int) -> tuple[int, bytes]:
    return 1, struct.pack("<I", _checked_int(value, "metadata value", 0, 0xFFFFFFFF))


def _meta_f32(value: float) -> tuple[int, bytes]:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("metadata float must be finite")
    return 4, struct.pack("<f", value)


def _meta_utf8(value: str) -> tuple[int, bytes]:
    encoded = value.encode("utf-8")
    if b"\0" in encoded:
        raise ValueError("metadata string cannot contain NUL")
    return 6, encoded


def _write_vad_espm_v3(
    output_path: Path,
    *,
    model_name: str,
    version_str: str,
    variant: str,
    extracted_layers: dict[str, dict[str, Any]],
    sparse_mel: list[dict[str, Any]],
    prior_bias_shift: float,
    checkpoint_sha256: str,
    mel_filterbank_sha256: str,
) -> dict[str, Any]:
    layer_specs = _MODEL_LAYERS.get(variant)
    if layer_specs is None or list(extracted_layers) != list(layer_specs):
        raise ValueError("model variant has an invalid or incomplete layer set")
    if len(sparse_mel) != FRONTEND_CONFIG["num_mels"]:
        raise ValueError("expected exactly 64 sparse mel filters")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("invalid checkpoint SHA-256")
    if not isinstance(mel_filterbank_sha256, str) or len(mel_filterbank_sha256) != 64:
        raise ValueError("invalid mel-filterbank SHA-256")
    if any(ch not in "0123456789abcdefABCDEF"
           for ch in checkpoint_sha256 + mel_filterbank_sha256):
        raise ValueError("SHA-256 values must be hexadecimal")
    if not math.isfinite(float(prior_bias_shift)):
        raise ValueError("prior_bias_shift must be finite")

    tensor_data = bytearray()
    tensor_records = []
    tensor_names = set()
    total_params = 0
    total_elements = 0
    total_data_bytes = 0

    def add_tensor(name: str, shape: tuple[int, ...], payload: bytes) -> None:
        nonlocal total_elements, total_data_bytes
        if name in tensor_names:
            raise ValueError(f"duplicate tensor name {name}")
        if not 1 <= len(shape) <= 8 or any(dimension <= 0 for dimension in shape):
            raise ValueError(f"invalid shape for tensor {name}")
        elements = math.prod(shape)
        if elements > 0xFFFFFFFF or len(payload) != elements * 4:
            raise ValueError(f"invalid FP32 payload size for tensor {name}")
        tensor_data.extend(bytes((-len(tensor_data)) % 16))
        data_offset = len(tensor_data)
        tensor_data.extend(payload)
        dimensions = shape + (0,) * (8 - len(shape))
        name_bytes = _fixed_string(name, 64, "tensor name")
        name_hash = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
        tensor_records.append(V3_TENSOR.pack(
            name_hash, 0, 2, len(shape), 0, 0, elements, *dimensions,
            0, 0xFF, bytes(3), 0, 0, data_offset, len(payload),
            name_bytes, bytes(12),
        ))
        tensor_names.add(name)
        total_elements += elements
        total_data_bytes += len(payload)

    layers_config = []
    for name, (expected_shape, bias_count) in layer_specs.items():
        layer = extracted_layers[name]
        shape = tuple(_checked_int(size, f"{name}.shape", 1, 0xFFFFFFFF)
                      for size in layer["w_shape"])
        if shape != expected_shape:
            raise ValueError(f"invalid shape for PulseVAD layer {name}")
        weight_bytes = _float32_bytes(_flatten(layer["weight"]), f"{name}.weight")
        expected_weight_count = math.prod(expected_shape)
        if len(weight_bytes) != expected_weight_count * 4:
            raise ValueError(f"invalid weight payload size for PulseVAD layer {name}")
        add_tensor(f"layer.{name}.weight", expected_shape, weight_bytes)
        total_params += expected_weight_count

        bias = layer["bias"]
        if bias_count is None:
            if bias is not None:
                raise ValueError(f"PulseVAD layer {name} must not have a bias")
            actual_bias_count = 0
        else:
            if bias is None:
                raise ValueError(f"PulseVAD layer {name} is missing its bias")
            bias_bytes = _float32_bytes(_flatten(bias), f"{name}.bias")
            if len(bias_bytes) != bias_count * 4:
                raise ValueError(f"invalid bias payload size for PulseVAD layer {name}")
            add_tensor(f"layer.{name}.bias", (bias_count,), bias_bytes)
            actual_bias_count = bias_count
            total_params += bias_count
        layers_config.append({
            "name": name,
            "shape": list(expected_shape),
            "weight_count": expected_weight_count,
            "bias_count": actual_bias_count,
        })

    mel_records = []
    for index, mel_bin in enumerate(sparse_mel):
        start = _checked_int(mel_bin["start_k"], f"mel_bins[{index}].start_k", 0, 256)
        count = _checked_int(mel_bin["num_k"], f"mel_bins[{index}].count", 0, 257)
        if start + count > 257:
            raise ValueError(f"invalid sparse mel-filter range at index {index}")
        values = mel_bin["weights"]
        if len(values) != count:
            raise ValueError(f"mel-filter weight count mismatch at index {index}")
        if count:
            mel_bytes = _float32_bytes(values, f"mel_bins[{index}]")
            add_tensor(f"mel.filter.{index:02d}.weights", (count,), mel_bytes)
        mel_records.append({"start_k": start, "count": count})

    metadata_records = {
        "general.architecture": _meta_utf8("pulsevad.cnn"),
        "general.architecture_version": _meta_u32(1),
        "pulsevad.variant": _meta_utf8(variant),
        "pulsevad.total_params": _meta_u32(total_params),
        "pulsevad.prior_bias_shift": _meta_f32(prior_bias_shift),
        "pulsevad.checkpoint_sha256": _meta_utf8(checkpoint_sha256.lower()),
        "pulsevad.mel_filterbank_sha256": _meta_utf8(mel_filterbank_sha256.lower()),
        "pulsevad.mel_filter_count": _meta_u32(FRONTEND_CONFIG["num_mels"]),
    }
    for key, value in FRONTEND_CONFIG.items():
        if isinstance(value, int):
            metadata_records[f"pulsevad.frontend.{key}"] = _meta_u32(value)
        else:
            metadata_records[f"pulsevad.frontend.{key}"] = _meta_f32(value)
    for index, record in enumerate(mel_records):
        metadata_records[f"pulsevad.mel.{index:02d}.start_k"] = _meta_u32(record["start_k"])
        metadata_records[f"pulsevad.mel.{index:02d}.count"] = _meta_u32(record["count"])

    metadata = _encode_v3_metadata(metadata_records)
    index = V3_INDEX_HEADER.pack(
        b"ETIX", 2, V3_INDEX_HEADER_SIZE, V3_TENSOR_RECORD_SIZE, 0,
        len(tensor_records), 1, total_elements, total_data_bytes,
    ) + b"".join(tensor_records)
    sections = [
        (SECTION_TENSOR_INDEX, 0, "tensor_index", index),
        (SECTION_TENSOR_DATA, 0, "tensor_data", bytes(tensor_data)),
        (SECTION_METADATA, 0, "metadata", metadata),
    ]
    if len(sections) > V3_MAX_SECTIONS:
        raise ValueError("too many ESPM sections")
    table_end = HEADER_SIZE + len(sections) * SECTION_ENTRY_SIZE
    offset = _align(table_end)
    rows = []
    for kind, section_flags, name, payload in sections:
        rows.append((kind, section_flags, name, offset, payload,
                     zlib.crc32(payload) & 0xFFFFFFFF))
        offset = _align(offset + len(payload))
    file_size = rows[-1][3] + len(rows[-1][4])
    section_table = b"".join(_SECTION.pack(
        kind, section_flags, section_offset, len(payload), checksum,
        _fixed_string(name, 16, "section name"),
    ) for kind, section_flags, name, section_offset, payload, checksum in rows)
    table_crc = zlib.crc32(section_table) & 0xFFFFFFFF
    psram_required_kb = (len(tensor_data) + 1023) // 1024

    def pack_header(header_crc: int) -> bytes:
        return _HEADER.pack(
            MAGIC, V3_FORMAT_VERSION, MODEL_TYPE_VAD, MODEL_QUANT_FP32,
            len(rows), _fixed_string(model_name, 32, "model name"),
            _fixed_string(version_str, 16, "version"),
            FRONTEND_CONFIG["sample_rate"], psram_required_kb, header_crc,
            file_size, HEADER_SIZE, HEADER_SIZE, SECTION_ENTRY_SIZE, ALIGNMENT,
            0, table_crc, bytes(28),
        )

    header = pack_header(0)
    header = pack_header(zlib.crc32(header) & 0xFFFFFFFF)
    output = bytearray(file_size)
    output[:HEADER_SIZE] = header
    output[HEADER_SIZE:HEADER_SIZE + len(section_table)] = section_table
    for _, _, _, section_offset, payload, _ in rows:
        output[section_offset:section_offset + len(payload)] = payload
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    return {
        "file_size": file_size,
        "parameter_count": total_params,
        "weights_size": len(tensor_data),
        "model_name": model_name,
    }


def write_vad_espm(
    output_path: Path,
    *,
    model_name: str,
    version_str: str,
    variant: str,
    extracted_layers: dict[str, dict[str, Any]],
    sparse_mel: list[dict[str, Any]],
    prior_bias_shift: float,
    checkpoint_sha256: str,
    mel_filterbank_sha256: str,
) -> dict[str, Any]:
    """Write a version 3 PulseVAD container with typed metadata and FP32 tensors."""
    return _write_vad_espm_v3(
        output_path,
        model_name=model_name,
        version_str=version_str,
        variant=variant,
        extracted_layers=extracted_layers,
        sparse_mel=sparse_mel,
        prior_bias_shift=prior_bias_shift,
        checkpoint_sha256=checkpoint_sha256,
        mel_filterbank_sha256=mel_filterbank_sha256,
    )


def _decode_v3_metadata(data: bytes) -> dict[str, tuple[int, Any, int]]:
    if len(data) < V3_METADATA_HEADER.size or len(data) > 64 * 1024:
        raise ValueError("invalid typed metadata size")
    magic, version, header_size, count, reserved = V3_METADATA_HEADER.unpack_from(data)
    if (magic != b"EMET" or version != 1 or header_size != 16 or
            count > 512 or reserved != 0):
        raise ValueError("unsupported typed metadata header")
    position = V3_METADATA_HEADER.size
    values = {}
    for _ in range(count):
        if len(data) - position < V3_METADATA_RECORD.size:
            raise ValueError("truncated metadata record")
        key_size, value_type, flags, value_size = V3_METADATA_RECORD.unpack_from(
            data, position)
        position += V3_METADATA_RECORD.size
        end = position + key_size + value_size
        if not key_size or flags & ~1 or end > len(data):
            raise ValueError("invalid metadata record size or flags")
        try:
            key = data[position:position + key_size].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("invalid metadata key UTF-8") from error
        if "\0" in key or key in values:
            raise ValueError("invalid or duplicate metadata key")
        position += key_size
        raw_value = data[position:position + value_size]
        position += value_size
        if value_type == 1 and value_size == 4:
            value = struct.unpack("<I", raw_value)[0]
        elif value_type == 2 and value_size == 8:
            value = struct.unpack("<Q", raw_value)[0]
        elif value_type == 3 and value_size == 4:
            value = struct.unpack("<i", raw_value)[0]
        elif value_type == 4 and value_size == 4:
            value = struct.unpack("<f", raw_value)[0]
            if not math.isfinite(value):
                raise ValueError("non-finite metadata float")
        elif value_type == 5:
            if value_size != 1 or raw_value[0] > 1:
                raise ValueError("invalid metadata boolean")
            value = bool(raw_value[0])
        elif value_type == 6:
            try:
                value = raw_value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("invalid metadata UTF-8") from error
            if "\0" in value:
                raise ValueError("metadata string contains NUL")
        elif value_type == 7:
            value = raw_value
        elif value_type in (1, 2, 3, 4):
            raise ValueError("invalid metadata value length")
        elif flags & 1:
            raise ValueError("unsupported required metadata type")
        else:
            value = raw_value
        values[key] = (value_type, value, flags)
    if position != len(data):
        raise ValueError("trailing bytes in typed metadata")
    return values


def _read_vad_espm_v3(container: bytes) -> dict[str, Any]:
    if len(container) < HEADER_SIZE:
        raise ValueError("ESPM file is shorter than its header")
    fields = _HEADER.unpack_from(container)
    (magic, version, model_type, quant_type, section_count, name_bytes, version_bytes,
     sample_rate, psram_kb, header_crc, file_size, table_offset, header_size,
     section_size, alignment, flags, table_crc, reserved) = fields
    if (magic != MAGIC or version != V3_FORMAT_VERSION or model_type != MODEL_TYPE_VAD or
            quant_type != MODEL_QUANT_FP32 or not 1 <= section_count <= V3_MAX_SECTIONS or
            file_size != len(container) or table_offset != HEADER_SIZE or
            header_size != HEADER_SIZE or section_size != SECTION_ENTRY_SIZE or
            alignment != ALIGNMENT or flags != 0 or reserved != bytes(28) or
            sample_rate != FRONTEND_CONFIG["sample_rate"] or psram_kb == 0):
        raise ValueError("invalid or unsupported ESPM v3 PulseVAD header")
    header_copy = bytearray(container[:HEADER_SIZE])
    header_copy[68:72] = bytes(4)
    if zlib.crc32(header_copy) & 0xFFFFFFFF != header_crc:
        raise ValueError("ESPM header CRC mismatch")
    model_name = _read_fixed_string(name_bytes, "model name")
    version_str = _read_fixed_string(version_bytes, "version")
    table_end = table_offset + section_count * section_size
    if table_end > len(container):
        raise ValueError("truncated ESPM section table")
    table = container[table_offset:table_end]
    if zlib.crc32(table) & 0xFFFFFFFF != table_crc:
        raise ValueError("ESPM section-table CRC mismatch")

    sections = {}
    names = set()
    ranges = []
    for index in range(section_count):
        kind, section_flags, offset, size, checksum, raw_name = _SECTION.unpack_from(
            table, index * section_size)
        name = _read_fixed_string(raw_name, "section name")
        if (section_flags & ~V3_SECTION_OPTIONAL or not size or offset % ALIGNMENT or
                offset < _align(table_end) or offset > len(container) or
                size > len(container) - offset or kind in sections or name in names):
            raise ValueError("invalid ESPM section record")
        if kind not in (SECTION_CONFIG_JSON, SECTION_TENSOR_INDEX,
                        SECTION_TENSOR_DATA, SECTION_METADATA):
            if not section_flags & V3_SECTION_OPTIONAL:
                raise ValueError("unknown required ESPM section")
        if kind == SECTION_CONFIG_JSON and not section_flags:
            raise ValueError("diagnostic JSON section must be optional")
        if kind in (SECTION_TENSOR_INDEX, SECTION_TENSOR_DATA, SECTION_METADATA) and section_flags:
            raise ValueError("required PulseVAD section is marked optional")
        payload = container[offset:offset + size]
        if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
            raise ValueError("ESPM section CRC mismatch")
        sections[kind] = payload
        names.add(name)
        ranges.append((offset, offset + size))
    ranges.sort()
    if (any(ranges[i][0] < ranges[i - 1][1] for i in range(1, len(ranges))) or
            max(end for _, end in ranges) != len(container)):
        raise ValueError("overlapping or trailing ESPM section data")
    required = {SECTION_TENSOR_INDEX, SECTION_TENSOR_DATA, SECTION_METADATA}
    if not required.issubset(sections):
        raise ValueError("missing required PulseVAD v3 section")

    tensor_data = sections[SECTION_TENSOR_DATA]
    if psram_kb * 1024 < len(tensor_data):
        raise ValueError("ESPM PSRAM lower bound is smaller than tensor data")

    metadata = _decode_v3_metadata(sections[SECTION_METADATA])
    frontend = {}
    required_metadata = {
        "general.architecture": (6, "pulsevad.cnn"),
        "general.architecture_version": (1, 1),
        "pulsevad.variant": (6, None),
        "pulsevad.total_params": (1, None),
        "pulsevad.prior_bias_shift": (4, None),
        "pulsevad.checkpoint_sha256": (6, None),
        "pulsevad.mel_filterbank_sha256": (6, None),
        "pulsevad.mel_filter_count": (1, FRONTEND_CONFIG["num_mels"]),
    }
    known_required = set(required_metadata)
    known_required.update(f"pulsevad.frontend.{key}" for key in FRONTEND_CONFIG)
    for index in range(FRONTEND_CONFIG["num_mels"]):
        known_required.add(f"pulsevad.mel.{index:02d}.start_k")
        known_required.add(f"pulsevad.mel.{index:02d}.count")
    if any(value[2] & 1 and key not in known_required
           for key, value in metadata.items()):
        raise ValueError("unknown required metadata key")
    for key, (expected_type, expected_value) in required_metadata.items():
        if key not in metadata or metadata[key][0] != expected_type or not metadata[key][2] & 1:
            raise ValueError(f"missing or invalid required metadata {key}")
        if expected_value is not None and metadata[key][1] != expected_value:
            raise ValueError(f"unsupported metadata value for {key}")
    variant = metadata["pulsevad.variant"][1]
    if variant not in _MODEL_LAYERS:
        raise ValueError("unsupported PulseVAD variant")
    for key, expected_value in FRONTEND_CONFIG.items():
        qualified = f"pulsevad.frontend.{key}"
        expected_type = 1 if isinstance(expected_value, int) else 4
        if (qualified not in metadata or metadata[qualified][0] != expected_type or
                not metadata[qualified][2] & 1):
            raise ValueError(f"missing or invalid frontend metadata {key}")
        value = metadata[qualified][1]
        if isinstance(expected_value, float):
            if abs(value - expected_value) >= (1e-7 if key == "preemphasis_alpha" else 1e-9):
                raise ValueError(f"unsupported frontend value {key}")
        elif value != expected_value:
            raise ValueError(f"unsupported frontend value {key}")
        frontend[key] = value
    for key in ("pulsevad.checkpoint_sha256", "pulsevad.mel_filterbank_sha256"):
        value = metadata[key][1]
        if len(value) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
            raise ValueError(f"invalid {key}")
    index = sections[SECTION_TENSOR_INDEX]
    if len(index) < V3_INDEX_HEADER_SIZE:
        raise ValueError("truncated ETIX v2 header")
    (index_magic, index_version, index_header_size, record_size, index_flags,
     tensor_count, scope_count, total_elements, total_data_bytes) = V3_INDEX_HEADER.unpack_from(index)
    if (index_magic != b"ETIX" or index_version != 2 or
            index_header_size != V3_INDEX_HEADER_SIZE or
            record_size != V3_TENSOR_RECORD_SIZE or index_flags != 0 or
            tensor_count == 0 or tensor_count > V3_MAX_TENSORS or scope_count != 1 or
            len(index) != V3_INDEX_HEADER_SIZE + tensor_count * V3_TENSOR_RECORD_SIZE):
        raise ValueError("invalid ETIX v2 index header")

    tensors = {}
    payload_ranges = []
    computed_elements = 0
    computed_bytes = 0
    for item in range(tensor_count):
        record = V3_TENSOR.unpack_from(
            index, V3_INDEX_HEADER_SIZE + item * V3_TENSOR_RECORD_SIZE)
        (name_hash, scope, dtype, rank, quant, layout, elements, *tail) = record
        shape = tuple(tail[:8])
        channels, axis, reserved, scales_offset, scales_size, data_offset, \
            data_size, raw_name, reserved_v2 = tail[8:]
        name = _read_fixed_string(raw_name, "tensor name")
        if (scope != 0 or dtype != 2 or quant != 0 or layout != 0 or
                not 1 <= rank <= 8 or any(dim <= 0 for dim in shape[:rank]) or
                any(shape[rank:]) or math.prod(shape[:rank]) != elements or
                channels != 0 or axis != 0xFF or scales_offset != 0 or scales_size != 0 or
                reserved != bytes(3) or reserved_v2 != bytes(12) or
                zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF != name_hash):
            raise ValueError("invalid FP32 tensor record")
        data_end = data_offset + data_size
        if (data_offset % 16 or data_end > len(tensor_data) or
                data_size != elements * 4 or name in tensors):
            raise ValueError("tensor payload is outside tensor data section")
        payload = tensor_data[data_offset:data_end]
        if any(not math.isfinite(value) for (value,) in struct.iter_unpack("<f", payload)):
            raise ValueError("tensor payload contains a non-finite value")
        tensors[name] = (shape[:rank], data_offset, elements, payload)
        payload_ranges.append((data_offset, data_end))
        computed_elements += elements
        computed_bytes += data_size
    payload_ranges.sort()
    if (any(payload_ranges[i][0] < payload_ranges[i - 1][1]
            for i in range(1, len(payload_ranges))) or
            computed_elements != total_elements or computed_bytes != total_data_bytes):
        raise ValueError("tensor payload ranges or ETIX totals are invalid")

    config = {
        "schema": "pulsevad",
        "schema_version": 1,
        "variant": variant,
        "frontend": frontend,
        "prior_bias_shift": metadata["pulsevad.prior_bias_shift"][1],
        "total_params": metadata["pulsevad.total_params"][1],
        "checkpoint_sha256": metadata["pulsevad.checkpoint_sha256"][1],
        "mel_filterbank_sha256": metadata["pulsevad.mel_filterbank_sha256"][1],
        "layers": [],
        "mel_bins": [],
    }
    used = set()
    parameter_count = 0
    for name, (expected_shape, bias_count) in _MODEL_LAYERS[variant].items():
        tensor_name = f"layer.{name}.weight"
        if tensor_name not in tensors or tensors[tensor_name][0] != expected_shape:
            raise ValueError(f"missing or invalid tensor {tensor_name}")
        _, weight_offset, weight_count, _ = tensors[tensor_name]
        used.add(tensor_name)
        parameter_count += weight_count
        layer = {
            "name": name,
            "shape": list(expected_shape),
            "weight_offset": weight_offset,
            "weight_count": weight_count,
        }
        bias_name = f"layer.{name}.bias"
        if bias_count is None:
            if bias_name in tensors:
                raise ValueError(f"unexpected tensor {bias_name}")
            layer.update({"bias_offset": None, "bias_count": 0})
        else:
            if bias_name not in tensors or tensors[bias_name][0] != (bias_count,):
                raise ValueError(f"missing or invalid tensor {bias_name}")
            _, bias_offset, _, _ = tensors[bias_name]
            used.add(bias_name)
            layer.update({"bias_offset": bias_offset, "bias_count": bias_count})
            parameter_count += bias_count
        config["layers"].append(layer)

    for index in range(FRONTEND_CONFIG["num_mels"]):
        start_key = f"pulsevad.mel.{index:02d}.start_k"
        count_key = f"pulsevad.mel.{index:02d}.count"
        if (start_key not in metadata or count_key not in metadata or
                metadata[start_key][0] != 1 or metadata[count_key][0] != 1 or
                not metadata[start_key][2] & 1 or not metadata[count_key][2] & 1):
            raise ValueError(f"missing sparse mel metadata {index}")
        start = metadata[start_key][1]
        count = metadata[count_key][1]
        if start > 256 or count > 257 or start + count > 257:
            raise ValueError(f"invalid sparse mel metadata {index}")
        mel = {"start_k": start, "count": count, "weights_offset": None}
        tensor_name = f"mel.filter.{index:02d}.weights"
        if count:
            if tensor_name not in tensors or tensors[tensor_name][0] != (count,):
                raise ValueError(f"missing sparse mel tensor {index}")
            _, offset, _, _ = tensors[tensor_name]
            used.add(tensor_name)
            mel["weights_offset"] = offset
        elif tensor_name in tensors:
            raise ValueError(f"unexpected empty sparse mel tensor {index}")
        config["mel_bins"].append(mel)
    if used != set(tensors) or parameter_count != config["total_params"]:
        raise ValueError("PulseVAD tensor inventory or parameter count mismatch")
    return {
        "model_name": model_name,
        "version": version_str,
        "config": config,
        "weights": tensor_data,
        "typed_metadata": {key: value[1] for key, value in metadata.items()},
    }


def read_vad_espm(container: bytes) -> dict[str, Any]:
    """Validate an ESPM v3 PulseVAD file."""
    if len(container) < HEADER_SIZE:
        raise ValueError("ESPM file is shorter than its header")
    version = struct.unpack_from("<H", container, 4)[0]
    if version != V3_FORMAT_VERSION:
        raise ValueError("unsupported ESPM format version")
    return _read_vad_espm_v3(container)
