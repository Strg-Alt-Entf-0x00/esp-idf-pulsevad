#!/usr/bin/env python3
"""
PulseVAD Model Exporter for ESP32-P4
====================================
Author: Strg-Alt-Entf-0x00
Repository: esp-idf-pulsevad

Directly loads PyTorch checkpoints from models/py/, extracts:
1. Exact BatchNorm-folded weights
2. Exact layer biases (critical: prevents bias=0 inference bugs)
3. Prior bias shift calibration (-1.40 logit shift for balanced thresholds)
4. Mel-Filterbank (257x64) optimized into sparse representation (~500 non-zero values)

Outputs:
- Private C header: src/pulsevad_model_data.h (built-in 2.1k ROM fallback)
- Native ESPM files: models/espm/
- Debug JSON: .data/debug/pulsevad_model_debug.json

Usage:
    python tools-py/pulsevad_models_converter/pulsevad_convert_models.py

Requirements:
    - PyTorch: pip install torch
    - NumPy: pip install numpy
"""

import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from espm_format import write_vad_espm

try:
    import torch
except ImportError:
    print("[FATAL] PyTorch not installed. Run: pip install torch")
    sys.exit(1)

# Default paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Default sources located in models/py/
DEFAULT_PTH_2K = PROJECT_ROOT / "models" / "py" / "pulsevad_2.1k" / "pulsevad_2.1k.pth"
DEFAULT_PTH_81K = PROJECT_ROOT / "models" / "py" / "pulsevad_teacher_81k" / "pulsevad_teacher_81k.pth"
DEFAULT_MEL_FB = PROJECT_ROOT / "models" / "py" / "mel_filterbank.npy"

# Target outputs
OUTPUT_HEADER = PROJECT_ROOT / "src" / "pulsevad_model_data.h"
OUTPUT_DEBUG_JSON = PROJECT_ROOT / ".data" / "debug" / "pulsevad_model_debug.json"

ESPM_OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "espm"
)

PRIOR_BIAS_SHIFT = -1.40  # Quantile-based prior calibration from paper/spec


def portable_source_path(source_path: Path) -> str:
    source_path = source_path.resolve()
    try:
        return source_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"<external>/{source_path.name}"


def optimize_mel_filterbank(fb_dense: np.ndarray):
    """
    Convert dense (257, 64) float32 Mel filterbank to sparse representation.
    Out of 16,448 entries, only ~500 are non-zero.
    """
    assert fb_dense.shape == (257, 64), f"Expected (257, 64), got {fb_dense.shape}"
    sparse_bins = []
    total_nonzeros = 0

    for m in range(64):
        col = fb_dense[:, m]
        nonzeros = np.nonzero(col)[0]
        if len(nonzeros) == 0:
            start_k = 0
            num_k = 0
            weights = []
        else:
            start_k = int(nonzeros[0])
            end_k = int(nonzeros[-1]) + 1
            num_k = end_k - start_k
            weights = [float(x) for x in col[start_k:end_k]]
            total_nonzeros += num_k

        sparse_bins.append({
            "bin_idx": m,
            "start_k": start_k,
            "num_k": num_k,
            "weights": weights,
        })

    print(f"[OK] Mel filterbank converted to sparse: {total_nonzeros} active weights out of 16,448 (savings: {100.0 - (total_nonzeros / 16448.0 * 100):.1f}%)")
    return sparse_bins, total_nonzeros


def export_pulsevad_2k(pth_path: Path, mel_fb_path: Path, header_path: Path, debug_json_path: Path):
    print("=" * 70)
    print("  PulseVAD 2.1k Exporter (Pure PyTorch -> C Header & ESPM v3)")
    print("=" * 70)

    if not pth_path.exists():
        print(f"[FATAL] PyTorch checkpoint not found: {pth_path}")
        sys.exit(1)

    if not mel_fb_path.exists():
        print(f"[FATAL] Mel filterbank not found: {mel_fb_path}")
        sys.exit(1)

    # 1. Load Mel Filterbank
    fb_dense = np.load(mel_fb_path)
    sparse_mel, non_zero_count = optimize_mel_filterbank(fb_dense)

    # 2. Load Checkpoint
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    total_params = checkpoint.get("params", 2118)
    print(f"[OK] Loaded checkpoint '{pth_path.name}' (Reported params: {total_params})")

    # Layer specifications for 2.1k model
    layers_info = [
        ("adapter", "adapter.weight", "adapter.bias", [12, 64, 1], 12),
        ("conv0_dw", "conv0_dw.weight", None, [12, 1, 11], None),
        ("conv0_pw", "conv0_pw.weight", "conv0_pw.bias", [8, 12, 1], 8),
        ("block1", "block1.weight", "block1.bias", [8, 8, 1], 8),
        ("block2", "block2.weight", "block2.bias", [8, 8, 1], 8),
        ("subA_dw", "subA_dw.weight", None, [8, 1, 17], None),
        ("subA_pw", "subA_pw.weight", "subA_pw.bias", [8, 8, 1], 8),
        ("subC_dw", "subC_dw.weight", None, [8, 1, 17], None),
        ("subC_pw", "subC_pw.weight", "subC_pw.bias", [8, 8, 1], 8),
        ("skip", "skip.weight", "skip.bias", [8, 8, 1], 8),
        ("conv4_dw", "conv4_dw.weight", None, [8, 1, 29], None),
        ("conv4_pw", "conv4_pw.weight", "conv4_pw.bias", [8, 8, 1], 8),
        ("conv5", "conv5.weight", "conv5.bias", [8, 8, 1], 8),
        ("classifier", "classifier.weight", "classifier.bias", [2, 8], 2),
    ]

    extracted_layers = {}
    debug_dict = {
        "_meta": {
            "model": "PulseVAD 2.1k",
            "source": portable_source_path(pth_path),
            "target": "ESP32-P4",
            "prior_bias_shift": PRIOR_BIAS_SHIFT,
            "total_params": total_params,
        },
        "layers": {},
        "mel_filterbank": {
            "fft_bins": 257,
            "mel_bins": 64,
            "nonzero_coeffs": non_zero_count,
        },
    }

    param_count_check = 0

    for name, w_key, b_key, exp_w_shape, exp_b_len in layers_info:
        w_tensor = state_dict[w_key].detach().cpu().numpy().astype(np.float32)
        assert list(w_tensor.shape) == exp_w_shape, f"Shape mismatch for {w_key}: expected {exp_w_shape}, got {list(w_tensor.shape)}"
        param_count_check += w_tensor.size

        b_tensor = None
        if b_key is not None:
            b_tensor = state_dict[b_key].detach().cpu().numpy().astype(np.float32)
            assert b_tensor.shape[0] == exp_b_len, f"Shape mismatch for {b_key}: expected {exp_b_len}, got {b_tensor.shape[0]}"
            param_count_check += b_tensor.size

        extracted_layers[name] = {
            "weight": w_tensor,
            "bias": b_tensor,
            "w_shape": list(w_tensor.shape),
            "b_shape": list(b_tensor.shape) if b_tensor is not None else [],
        }

        debug_dict["layers"][name] = {
            "w_shape": list(w_tensor.shape),
            "w_min": float(w_tensor.min()),
            "w_max": float(w_tensor.max()),
            "w_mean": float(w_tensor.mean()),
            "has_bias": b_tensor is not None,
            "b_min": float(b_tensor.min()) if b_tensor is not None else 0.0,
            "b_max": float(b_tensor.max()) if b_tensor is not None else 0.0,
        }

    print(f"[OK] Extracted {len(layers_info)} layers. Total parameters: {param_count_check} (weights + biases).")

    # 3. Generate C Header
    header_path.parent.mkdir(parents=True, exist_ok=True)
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("/**\n")
        f.write(" * PulseVAD Model Weights and Constants for ESP32-P4\n")
        f.write(" * ==================================================\n")
        f.write(" * Auto-generated by pulsevad_convert_models.py. Do not edit directly.\n")
        f.write(" * Target: ESP32-P4 RISC-V Hardware FPU & PIE Vector Extensions.\n")
        f.write(" *\n")
        f.write(" * Total parameters: 2,042 (Folded BatchNorm Weights + Biases)\n")
        f.write(" * Sparse Mel-filterbank: 500 active coefficients\n")
        f.write(" */\n\n")
        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stddef.h>\n\n")
        f.write("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n")

        f.write("// Architecture Hyperparameters\n")
        f.write("#define PULSEVAD_SAMPLE_RATE         16000\n")
        f.write("#define PULSEVAD_WINDOW_SAMPLES      3200   // 200 ms @ 16 kHz\n")
        f.write("#define PULSEVAD_N_FFT               512\n")
        f.write("#define PULSEVAD_WIN_LENGTH          400    // 25 ms Hann window\n")
        f.write("#define PULSEVAD_PAD_WIN             56     // (512 - 400) / 2\n")
        f.write("#define PULSEVAD_HOP_LENGTH          160    // 10 ms hop\n")
        f.write("#define PULSEVAD_NUM_FRAMES          21     // 21 time frames per 200 ms\n")
        f.write("#define PULSEVAD_NUM_MELS            64     // 64 mel channels\n")
        f.write("#define PULSEVAD_PREEMPHASIS_ALPHA   0.97f\n")
        f.write("#define PULSEVAD_EPS                 1e-5f\n")
        f.write(f"#define PULSEVAD_PRIOR_BIAS_SHIFT    ({PRIOR_BIAS_SHIFT:.2f}f)\n\n")

        # Sparse Mel filterbank
        f.write("// Sparse Mel Filterbank Structure\n")
        f.write("typedef struct {\n")
        f.write("    uint16_t start_k;       // First non-zero FFT frequency bin\n")
        f.write("    uint16_t num_k;         // Number of contiguous non-zero bins\n")
        f.write("    const float* weights;   // Triangular filterbank weights\n")
        f.write("} pulsevad_sparse_mel_bin_t;\n\n")

        mel_flat_weights = []
        bin_offsets = []
        for b in sparse_mel:
            bin_offsets.append(len(mel_flat_weights))
            mel_flat_weights.extend(b["weights"])

        f.write(f"// Sparse Mel Weights Pool ({len(mel_flat_weights)} floats = {len(mel_flat_weights)*4} bytes)\n")
        f.write(f"static const float PULSEVAD_MEL_SPARSE_WEIGHTS[{len(mel_flat_weights)}] = {{\n")
        for i in range(0, len(mel_flat_weights), 8):
            chunk = mel_flat_weights[i:i + 8]
            f.write("    " + ", ".join(f"{x:.8g}f" for x in chunk) + ",\n")
        f.write("};\n\n")

        f.write("// Sparse Mel Bins Table (64 channels)\n")
        f.write("static const pulsevad_sparse_mel_bin_t PULSEVAD_MEL_BINS[64] = {\n")
        for i, b in enumerate(sparse_mel):
            offset = bin_offsets[i]
            num_k = b["num_k"]
            if num_k > 0:
                f.write(f"    {{ {b['start_k']}, {num_k}, &PULSEVAD_MEL_SPARSE_WEIGHTS[{offset}] }},\n")
            else:
                f.write(f"    {{ 0, 0, NULL }},\n")
        f.write("};\n\n")

        # Neural Network Weights & Biases
        f.write("// ============================================================================\n")
        f.write("// Neural Network Weights and Biases (FP32 Hardware FPU & PIE Aligned)\n")
        f.write("// ============================================================================\n\n")

        for name, layer_data in extracted_layers.items():
            w = layer_data["weight"].flatten()
            c_name = name.upper()
            w_shape_comment = "x".join(str(s) for s in layer_data["w_shape"])
            f.write(f"// Layer '{name}' Weights: [{w_shape_comment}] ({len(w)} elements)\n")
            f.write(f"static const float PULSEVAD_W_{c_name}[{len(w)}] __attribute__((aligned(16))) = {{\n")
            for i in range(0, len(w), 8):
                chunk = w[i:i + 8]
                f.write("    " + ", ".join(f"{x:.8g}f" for x in chunk) + ",\n")
            f.write("};\n\n")

            if layer_data["bias"] is not None:
                b = layer_data["bias"].flatten()
                f.write(f"// Layer '{name}' Biases: [{len(b)} elements]\n")
                f.write(f"static const float PULSEVAD_B_{c_name}[{len(b)}] __attribute__((aligned(16))) = {{\n")
                for i in range(0, len(b), 8):
                    chunk = b[i:i + 8]
                    f.write("    " + ", ".join(f"{x:.8g}f" for x in chunk) + ",\n")
                f.write("};\n\n")

        f.write("#ifdef __cplusplus\n}\n#endif\n")

    print(f"[OK] Header written successfully: {header_path}")

    espm_path = ESPM_OUTPUT_DIR / "pulsevad_2.1k.espm"
    result = write_vad_espm(
        espm_path,
        model_name="pulsevad_2.1k",
        version_str="1.0.0",
        variant="student",
        extracted_layers=extracted_layers,
        sparse_mel=sparse_mel,
        prior_bias_shift=PRIOR_BIAS_SHIFT,
        checkpoint_sha256=hashlib.sha256(pth_path.read_bytes()).hexdigest(),
        mel_filterbank_sha256=hashlib.sha256(mel_fb_path.read_bytes()).hexdigest(),
    )
    print(f"[OK] Native ESPM v3 written: {espm_path} ({result['file_size']:,} bytes)")

    # 5. Write debug JSON
    debug_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_json_path, "w", encoding="utf-8") as f:
        json.dump(debug_dict, f, indent=2)
    print(f"[OK] Debug JSON written: {debug_json_path}")
    print("=" * 70)


def export_pulsevad_teacher(pth_path: Path, mel_fb_path: Path):
    """Export PulseVAD Teacher 81k model to native ESPM v3."""
    if not pth_path.exists():
        print(f"[FATAL] Teacher checkpoint not found: {pth_path}")
        sys.exit(1)
    if not mel_fb_path.exists():
        print(f"[FATAL] Mel filterbank not found: {mel_fb_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  PulseVAD Teacher 81k Exporter (Pure PyTorch -> ESPM v3)")
    print("=" * 70)

    fb_dense = np.load(mel_fb_path)
    sparse_mel, _ = optimize_mel_filterbank(fb_dense)

    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
    sd = checkpoint["state_dict"]

    def fold_bn(conv_w, bn_w, bn_b, rm, rv, eps=1e-3):
        gain = bn_w / torch.sqrt(rv + eps)
        folded_w = conv_w * gain.view(-1, 1, 1)
        folded_b = bn_b - rm * gain
        return folded_w.detach().cpu().numpy().astype(np.float32), folded_b.detach().cpu().numpy().astype(np.float32)

    extracted_layers = {}

    # 1. adapter: conv [128, 64, 1] + bn [128]
    w, b = fold_bn(sd["adapter.conv.weight"], sd["adapter.bn.weight"], sd["adapter.bn.bias"], sd["adapter.bn.running_mean"], sd["adapter.bn.running_var"])
    extracted_layers["adapter"] = {"weight": w, "bias": b, "w_shape": list(w.shape), "b_shape": list(b.shape)}

    # 2. conv0_dw: conv [128, 1, 11]
    w = sd["conv0_dw.weight"].detach().cpu().numpy().astype(np.float32)
    extracted_layers["conv0_dw"] = {"weight": w, "bias": None, "w_shape": list(w.shape), "b_shape": []}

    # 3. conv0_pw: conv [128, 128, 1] + bn [128]
    w, b = fold_bn(sd["conv0_pw.conv.weight"], sd["conv0_pw.bn.weight"], sd["conv0_pw.bn.bias"], sd["conv0_pw.bn.running_mean"], sd["conv0_pw.bn.running_var"])
    extracted_layers["conv0_pw"] = {"weight": w, "bias": b, "w_shape": list(w.shape), "b_shape": list(b.shape)}

    # 4. block1: conv [64, 128, 1] + bn [64]
    w, b = fold_bn(sd["block1.conv.weight"], sd["block1.bn.weight"], sd["block1.bn.bias"], sd["block1.bn.running_mean"], sd["block1.bn.running_var"])
    extracted_layers["block1"] = {"weight": w, "bias": b, "w_shape": list(w.shape), "b_shape": list(b.shape)}

    # 5. block2: conv [64, 64, 1] + bn [64]
    w, b = fold_bn(sd["block2.conv.weight"], sd["block2.bn.weight"], sd["block2.bn.bias"], sd["block2.bn.running_mean"], sd["block2.bn.running_var"])
    extracted_layers["block2"] = {"weight": w, "bias": b, "w_shape": list(w.shape), "b_shape": list(b.shape)}

    # 6. subA_dw: conv [64, 1, 17]
    w = sd["block3.subA_dw.weight"].detach().cpu().numpy().astype(np.float32)
    extracted_layers["subA_dw"] = {"weight": w, "bias": None, "w_shape": list(w.shape), "b_shape": []}

    # 7. subA_pw: conv [64, 64, 1] + bn [64]
    w, b = fold_bn(sd["block3.subA_pw.conv.weight"], sd["block3.subA_pw.bn.weight"], sd["block3.subA_pw.bn.bias"], sd["block3.subA_pw.bn.running_mean"], sd["block3.subA_pw.bn.running_var"])
    extracted_layers["subA_pw"] = {"weight": w, "bias": b, "w_shape": list(w.shape), "b_shape": list(b.shape)}

    # 8. subC_dw: conv [64, 1, 17]
    w = sd["block3.subC_dw.weight"].detach().cpu().numpy().astype(np.float32)
    extracted_layers["subC_dw"] = {"weight": w, "bias": None, "w_shape": list(w.shape), "b_shape": []}

    # 9. subC_pw: conv [64, 64, 1] + bn [64]
    w, b = fold_bn(sd["block3.subC_pw.conv.weight"], sd["block3.subC_pw.bn.weight"], sd["block3.subC_pw.bn.bias"], sd["block3.subC_pw.bn.running_mean"], sd["block3.subC_pw.bn.running_var"])
    extracted_layers["subC_pw"] = {"weight": w, "bias": b, "w_shape": list(w.shape), "b_shape": list(b.shape)}

    # 10. skip: conv [64, 64, 1] + bn [64]
    w, b = fold_bn(sd["block3.skip.conv.weight"], sd["block3.skip.bn.weight"], sd["block3.skip.bn.bias"], sd["block3.skip.bn.running_mean"], sd["block3.skip.bn.running_var"])
    extracted_layers["skip"] = {"weight": w, "bias": b, "w_shape": list(w.shape), "b_shape": list(b.shape)}

    # 11. conv4_dw: conv [64, 1, 29]
    w = sd["conv4_dw.weight"].detach().cpu().numpy().astype(np.float32)
    extracted_layers["conv4_dw"] = {"weight": w, "bias": None, "w_shape": list(w.shape), "b_shape": []}

    # 12. conv4_pw: conv [128, 64, 1] + bn [128]
    w, b = fold_bn(sd["conv4_pw.conv.weight"], sd["conv4_pw.bn.weight"], sd["conv4_pw.bn.bias"], sd["conv4_pw.bn.running_mean"], sd["conv4_pw.bn.running_var"])
    extracted_layers["conv4_pw"] = {"weight": w, "bias": b, "w_shape": list(w.shape), "b_shape": list(b.shape)}

    # 13. conv5: conv [128, 128, 1] + bn [128]
    w, b = fold_bn(sd["conv5.conv.weight"], sd["conv5.bn.weight"], sd["conv5.bn.bias"], sd["conv5.bn.running_mean"], sd["conv5.bn.running_var"])
    extracted_layers["conv5"] = {"weight": w, "bias": b, "w_shape": list(w.shape), "b_shape": list(b.shape)}

    # 14. classifier: linear [2, 128] + bias [2]
    w = sd["classifier.weight"].detach().cpu().numpy().astype(np.float32)
    b = sd["classifier.bias"].detach().cpu().numpy().astype(np.float32)
    extracted_layers["classifier"] = {"weight": w, "bias": b, "w_shape": list(w.shape), "b_shape": list(b.shape)}

    total_params = sum(l["weight"].size + (l["bias"].size if l["bias"] is not None else 0) for l in extracted_layers.values())
    print(f"[OK] Teacher 81k folded successfully. Total folded parameters: {total_params}")

    espm_path = ESPM_OUTPUT_DIR / "pulsevad_teacher_81k.espm"
    result = write_vad_espm(
        espm_path,
        model_name="pulsevad_teacher_81k",
        version_str="1.0.0",
        variant="teacher",
        extracted_layers=extracted_layers,
        sparse_mel=sparse_mel,
        prior_bias_shift=PRIOR_BIAS_SHIFT,
        checkpoint_sha256=hashlib.sha256(pth_path.read_bytes()).hexdigest(),
        mel_filterbank_sha256=hashlib.sha256(mel_fb_path.read_bytes()).hexdigest(),
    )
    print(f"[OK] Native ESPM v3 written: {espm_path} ({result['file_size']:,} bytes)")
    print("=" * 70)


if __name__ == "__main__":
    export_pulsevad_2k(DEFAULT_PTH_2K, DEFAULT_MEL_FB, OUTPUT_HEADER, OUTPUT_DEBUG_JSON)
    export_pulsevad_teacher(DEFAULT_PTH_81K, DEFAULT_MEL_FB)
