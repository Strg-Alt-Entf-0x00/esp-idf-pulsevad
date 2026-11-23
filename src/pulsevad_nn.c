/**
 * PulseVAD Neural Network Implementation for ESP32-P4
 * ====================================================
 * Depthwise-separable & dilated 1D CNN running on RISC-V Hardware FPU.
 * Total parameters: 2,042 (Folded BatchNorm Weights + Biases).
 */

#include "pulsevad_internal.h"
#include <string.h>
#include <math.h>

#define T_LEN PULSEVAD_NUM_FRAMES  // 21 time frames

// Helper for 1x1 pointwise convolution
static inline void conv1d_pointwise_single_output(
    const float *inp,
    int c_in,
    const float *w_row,
    float bias,
    bool relu,
    float *out_row
) {
    int t = 0;
    for (; t + 6 < T_LEN; t += 7) {
        float acc0 = bias;
        float acc1 = bias;
        float acc2 = bias;
        float acc3 = bias;
        float acc4 = bias;
        float acc5 = bias;
        float acc6 = bias;
        int i = 0;
        for (; i + 3 < c_in; i += 4) {
            const float w0 = w_row[i + 0];
            const float w1 = w_row[i + 1];
            const float w2 = w_row[i + 2];
            const float w3 = w_row[i + 3];
            acc0 += inp[(i + 0) * T_LEN + t + 0] * w0;
            acc0 += inp[(i + 1) * T_LEN + t + 0] * w1;
            acc0 += inp[(i + 2) * T_LEN + t + 0] * w2;
            acc0 += inp[(i + 3) * T_LEN + t + 0] * w3;
            acc1 += inp[(i + 0) * T_LEN + t + 1] * w0;
            acc1 += inp[(i + 1) * T_LEN + t + 1] * w1;
            acc1 += inp[(i + 2) * T_LEN + t + 1] * w2;
            acc1 += inp[(i + 3) * T_LEN + t + 1] * w3;
            acc2 += inp[(i + 0) * T_LEN + t + 2] * w0;
            acc2 += inp[(i + 1) * T_LEN + t + 2] * w1;
            acc2 += inp[(i + 2) * T_LEN + t + 2] * w2;
            acc2 += inp[(i + 3) * T_LEN + t + 2] * w3;
            acc3 += inp[(i + 0) * T_LEN + t + 3] * w0;
            acc3 += inp[(i + 1) * T_LEN + t + 3] * w1;
            acc3 += inp[(i + 2) * T_LEN + t + 3] * w2;
            acc3 += inp[(i + 3) * T_LEN + t + 3] * w3;
            acc4 += inp[(i + 0) * T_LEN + t + 4] * w0;
            acc4 += inp[(i + 1) * T_LEN + t + 4] * w1;
            acc4 += inp[(i + 2) * T_LEN + t + 4] * w2;
            acc4 += inp[(i + 3) * T_LEN + t + 4] * w3;
            acc5 += inp[(i + 0) * T_LEN + t + 5] * w0;
            acc5 += inp[(i + 1) * T_LEN + t + 5] * w1;
            acc5 += inp[(i + 2) * T_LEN + t + 5] * w2;
            acc5 += inp[(i + 3) * T_LEN + t + 5] * w3;
            acc6 += inp[(i + 0) * T_LEN + t + 6] * w0;
            acc6 += inp[(i + 1) * T_LEN + t + 6] * w1;
            acc6 += inp[(i + 2) * T_LEN + t + 6] * w2;
            acc6 += inp[(i + 3) * T_LEN + t + 6] * w3;
        }
        for (; i < c_in; i++) {
            const float weight = w_row[i];
            acc0 += inp[i * T_LEN + t + 0] * weight;
            acc1 += inp[i * T_LEN + t + 1] * weight;
            acc2 += inp[i * T_LEN + t + 2] * weight;
            acc3 += inp[i * T_LEN + t + 3] * weight;
            acc4 += inp[i * T_LEN + t + 4] * weight;
            acc5 += inp[i * T_LEN + t + 5] * weight;
            acc6 += inp[i * T_LEN + t + 6] * weight;
        }
        if (relu) {
            if (acc0 < 0.0f) {
                acc0 = 0.0f;
            }
            if (acc1 < 0.0f) {
                acc1 = 0.0f;
            }
            if (acc2 < 0.0f) {
                acc2 = 0.0f;
            }
            if (acc3 < 0.0f) {
                acc3 = 0.0f;
            }
            if (acc4 < 0.0f) {
                acc4 = 0.0f;
            }
            if (acc5 < 0.0f) {
                acc5 = 0.0f;
            }
            if (acc6 < 0.0f) {
                acc6 = 0.0f;
            }
        }
        out_row[t + 0] = acc0;
        out_row[t + 1] = acc1;
        out_row[t + 2] = acc2;
        out_row[t + 3] = acc3;
        out_row[t + 4] = acc4;
        out_row[t + 5] = acc5;
        out_row[t + 6] = acc6;
    }
    for (; t < T_LEN; t++) {
        float acc = bias;
        int i = 0;
        for (; i + 3 < c_in; i += 4) {
            acc += inp[(i + 0) * T_LEN + t] * w_row[i + 0];
            acc += inp[(i + 1) * T_LEN + t] * w_row[i + 1];
            acc += inp[(i + 2) * T_LEN + t] * w_row[i + 2];
            acc += inp[(i + 3) * T_LEN + t] * w_row[i + 3];
        }
        for (; i < c_in; i++) {
            acc += inp[i * T_LEN + t] * w_row[i];
        }
        if (relu && acc < 0.0f) {
            acc = 0.0f;
        }
        out_row[t] = acc;
    }
}

static inline void conv1d_pointwise_pair(
    const float *inp,
    int c_in,
    const float *w0,
    const float *w1,
    float bias0,
    float bias1,
    bool relu,
    float *out0,
    float *out1
) {
    int t = 0;
    for (; t + 6 < T_LEN; t += 7) {
        float acc0[7];
        float acc1[7];
        for (int frame = 0; frame < 7; frame++) {
            acc0[frame] = bias0;
            acc1[frame] = bias1;
        }

        int i = 0;
        for (; i + 3 < c_in; i += 4) {
            for (int lane = 0; lane < 4; lane++) {
                const int channel = i + lane;
                const float weight0 = w0[channel];
                const float weight1 = w1[channel];
                const float *input_row = &inp[channel * T_LEN + t];
                for (int frame = 0; frame < 7; frame++) {
                    const float value = input_row[frame];
                    acc0[frame] += value * weight0;
                    acc1[frame] += value * weight1;
                }
            }
        }
        for (; i < c_in; i++) {
            const float weight0 = w0[i];
            const float weight1 = w1[i];
            const float *input_row = &inp[i * T_LEN + t];
            for (int frame = 0; frame < 7; frame++) {
                const float value = input_row[frame];
                acc0[frame] += value * weight0;
                acc1[frame] += value * weight1;
            }
        }

        for (int frame = 0; frame < 7; frame++) {
            if (relu) {
                if (acc0[frame] < 0.0f) {
                    acc0[frame] = 0.0f;
                }
                if (acc1[frame] < 0.0f) {
                    acc1[frame] = 0.0f;
                }
            }
            out0[t + frame] = acc0[frame];
            out1[t + frame] = acc1[frame];
        }
    }

    for (; t < T_LEN; t++) {
        float acc0 = bias0;
        float acc1 = bias1;
        for (int i = 0; i < c_in; i++) {
            const float value = inp[i * T_LEN + t];
            acc0 += value * w0[i];
            acc1 += value * w1[i];
        }
        if (relu) {
            if (acc0 < 0.0f) {
                acc0 = 0.0f;
            }
            if (acc1 < 0.0f) {
                acc1 = 0.0f;
            }
        }
        out0[t] = acc0;
        out1[t] = acc1;
    }
}

static inline void conv1d_pointwise(
    const float *inp,
    int c_in,
    int c_out,
    const float *w,
    const float *b,
    bool relu,
    float *out
) {
    int o = 0;
    for (; o + 1 < c_out; o += 2) {
        const float *w0 = &w[o * c_in];
        const float *w1 = &w[(o + 1) * c_in];
        const float bias0 = b ? b[o] : 0.0f;
        const float bias1 = b ? b[o + 1] : 0.0f;
        float *out0 = &out[o * T_LEN];
        float *out1 = &out[(o + 1) * T_LEN];
        conv1d_pointwise_pair(inp, c_in, w0, w1, bias0, bias1, relu, out0, out1);
    }

    if (o < c_out) {
        const float *w_row = &w[o * c_in];
        const float bias = b ? b[o] : 0.0f;
        float *out_row = &out[o * T_LEN];
        conv1d_pointwise_single_output(inp, c_in, w_row, bias, relu, out_row);
    }
}

// Helper for depthwise convolution (per-channel kernel)
static inline void conv1d_depthwise(
    const float *inp,
    int channels,
    int k_size,
    int pad,
    int dilation,
    const float *w,
    float *out
) {
    int first_k[T_LEN];
    int end_k[T_LEN];
    for (int t = 0; t < T_LEN; t++) {
        const int before_start = pad - t;
        first_k[t] = before_start > 0
            ? (before_start + dilation - 1) / dilation
            : 0;
        end_k[t] = (T_LEN - 1 - t + pad) / dilation + 1;
        if (end_k[t] > k_size) {
            end_k[t] = k_size;
        }
    }

    for (int c = 0; c < channels; c++) {
        const float *inp_c = &inp[c * T_LEN];
        const float *k_c = &w[c * k_size];
        float *out_c = &out[c * T_LEN];

        for (int t = 0; t < T_LEN; t++) {
            float acc = 0.0f;
            for (int k = first_k[t]; k < end_k[t]; k++) {
                int in_idx = t - pad + k * dilation;
                acc += inp_c[in_idx] * k_c[k];
            }
            out_c[t] = acc;
        }
    }
}

float pulsevad_nn_forward(pulsevad_handle_t handle, const float *features_64x21) {
    // Determine which model to run (dynamic loaded model vs built-in Flash ROM 2.1k)
    pulsevad_dynamic_model_t *dm = handle ? handle->dynamic_model : NULL;

    // Allocate intermediate working buffers
    static float s_fallback_bufA[128 * T_LEN] __attribute__((aligned(16)));
    static float s_fallback_bufB[128 * T_LEN] __attribute__((aligned(16)));
    static float s_fallback_res[64 * T_LEN]   __attribute__((aligned(16)));
    static float s_fallback_main[64 * T_LEN]  __attribute__((aligned(16)));

    float *bufA = handle ? handle->nn_bufA : s_fallback_bufA;
    float *bufB = handle ? handle->nn_bufB : s_fallback_bufB;
    float *res  = handle ? handle->nn_res  : s_fallback_res;
    float *main_b = handle ? handle->nn_main : s_fallback_main;

    if (dm == NULL) {
        // ====================================================================
        // Mode A: Built-in PulseVAD 2.1k Flash ROM Model (Zero RAM, Fast FPU)
        // ====================================================================
        conv1d_pointwise(features_64x21, 64, 12, PULSEVAD_W_ADAPTER, PULSEVAD_B_ADAPTER, true, bufA);
        conv1d_depthwise(bufA, 12, 11, 5, 1, PULSEVAD_W_CONV0_DW, bufB);
        conv1d_pointwise(bufB, 12, 8, PULSEVAD_W_CONV0_PW, PULSEVAD_B_CONV0_PW, true, bufA);
        conv1d_pointwise(bufA, 8, 8, PULSEVAD_W_BLOCK1, PULSEVAD_B_BLOCK1, true, bufB);
        conv1d_pointwise(bufB, 8, 8, PULSEVAD_W_BLOCK2, PULSEVAD_B_BLOCK2, true, res);

        conv1d_depthwise(res, 8, 17, 8, 1, PULSEVAD_W_SUBA_DW, bufA);
        conv1d_pointwise(bufA, 8, 8, PULSEVAD_W_SUBA_PW, PULSEVAD_B_SUBA_PW, true, bufB);
        conv1d_depthwise(bufB, 8, 17, 8, 1, PULSEVAD_W_SUBC_DW, bufA);
        conv1d_pointwise(bufA, 8, 8, PULSEVAD_W_SUBC_PW, PULSEVAD_B_SUBC_PW, false, main_b);

        conv1d_pointwise(res, 8, 8, PULSEVAD_W_SKIP, PULSEVAD_B_SKIP, false, bufB);
        for (int i = 0; i < 8 * T_LEN; i++) {
            float sum = main_b[i] + bufB[i];
            bufA[i] = (sum > 0.0f) ? sum : 0.0f;
        }

        conv1d_depthwise(bufA, 8, 29, 28, 2, PULSEVAD_W_CONV4_DW, bufB);
        conv1d_pointwise(bufB, 8, 8, PULSEVAD_W_CONV4_PW, PULSEVAD_B_CONV4_PW, true, bufA);
        conv1d_pointwise(bufA, 8, 8, PULSEVAD_W_CONV5, PULSEVAD_B_CONV5, true, bufB);

        float gap[8];
        for (int c = 0; c < 8; c++) {
            float sum = 0.0f;
            const float *row = &bufB[c * T_LEN];
            for (int t = 0; t < T_LEN; t++) {
                sum += row[t];
            }
            gap[c] = sum / (float)T_LEN;
        }

        float logit_0 = PULSEVAD_B_CLASSIFIER[0];
        float logit_1 = PULSEVAD_B_CLASSIFIER[1];
        for (int c = 0; c < 8; c++) {
            logit_0 += gap[c] * PULSEVAD_W_CLASSIFIER[c];
            logit_1 += gap[c] * PULSEVAD_W_CLASSIFIER[8 + c];
        }

        float logit_diff = logit_1 - logit_0;
#if CONFIG_PULSEVAD_FAST_SIGMOID
        return 0.5f + 0.5f * (logit_diff / (1.0f + fabsf(logit_diff)));
#else
        return 1.0f / (1.0f + expf(-logit_diff));
#endif
    } else {
        // ====================================================================
        // Mode B: Dynamic ESPM model (PulseVAD student or teacher)
        // ====================================================================
        const int c_ad_out  = dm->adapter->w_shape[0];   // 12 (2k) or 128 (81k)
        const int c_c0_out  = dm->conv0_pw->w_shape[0];  // 8 (2k) or 128 (81k)
        const int c_b1_out  = dm->block1->w_shape[0];    // 8 (2k) or 64 (81k)
        const int c_b2_out  = dm->block2->w_shape[0];    // 8 (2k) or 64 (81k)
        const int c_sa_out  = dm->subA_pw->w_shape[0];   // 8 (2k) or 64 (81k)
        const int c_sc_out  = dm->subC_pw->w_shape[0];   // 8 (2k) or 64 (81k)
        const int c_c4_out  = dm->conv4_pw->w_shape[0];  // 8 (2k) or 128 (81k)
        const int c_c5_out  = dm->conv5->w_shape[0];     // 8 (2k) or 128 (81k)

        // 1. Adapter: 64 -> c_ad_out
        conv1d_pointwise(features_64x21, 64, c_ad_out, dm->adapter->weights, dm->adapter->biases, true, bufA);
        // 2. Conv0_dw: c_ad_out -> c_ad_out
        conv1d_depthwise(bufA, c_ad_out, 11, 5, 1, dm->conv0_dw->weights, bufB);
        // 3. Conv0_pw: c_ad_out -> c_c0_out
        conv1d_pointwise(bufB, c_ad_out, c_c0_out, dm->conv0_pw->weights, dm->conv0_pw->biases, true, bufA);
        // 4. Block1: c_c0_out -> c_b1_out
        conv1d_pointwise(bufA, c_c0_out, c_b1_out, dm->block1->weights, dm->block1->biases, true, bufB);
        // 5. Block2: c_b1_out -> c_b2_out (saved to res for skip)
        conv1d_pointwise(bufB, c_b1_out, c_b2_out, dm->block2->weights, dm->block2->biases, true, res);

        // 6. SubA_dw: c_b2_out -> c_b2_out
        conv1d_depthwise(res, c_b2_out, 17, 8, 1, dm->subA_dw->weights, bufA);
        // 7. SubA_pw: c_b2_out -> c_sa_out
        conv1d_pointwise(bufA, c_b2_out, c_sa_out, dm->subA_pw->weights, dm->subA_pw->biases, true, bufB);
        // 8. SubC_dw: c_sa_out -> c_sa_out
        conv1d_depthwise(bufB, c_sa_out, 17, 8, 1, dm->subC_dw->weights, bufA);
        // 9. SubC_pw: c_sa_out -> c_sc_out (linear)
        conv1d_pointwise(bufA, c_sa_out, c_sc_out, dm->subC_pw->weights, dm->subC_pw->biases, false, main_b);

        // 10. Skip: c_b2_out -> c_sc_out (linear)
        conv1d_pointwise(res, c_b2_out, c_sc_out, dm->skip->weights, dm->skip->biases, false, bufB);

        // 11. Residual add
        for (int i = 0; i < c_sc_out * T_LEN; i++) {
            float sum = main_b[i] + bufB[i];
            bufA[i] = (sum > 0.0f) ? sum : 0.0f;
        }

        // 12. Conv4_dw: c_sc_out -> c_sc_out
        conv1d_depthwise(bufA, c_sc_out, 29, 28, 2, dm->conv4_dw->weights, bufB);
        // 13. Conv4_pw: c_sc_out -> c_c4_out
        conv1d_pointwise(bufB, c_sc_out, c_c4_out, dm->conv4_pw->weights, dm->conv4_pw->biases, true, bufA);
        // 14. Conv5: c_c4_out -> c_c5_out
        conv1d_pointwise(bufA, c_c4_out, c_c5_out, dm->conv5->weights, dm->conv5->biases, true, bufB);

        // 15. Global Average Pooling (GAP) across the 21 time frames
        float gap[128];
        for (int c = 0; c < c_c5_out; c++) {
            float sum = 0.0f;
            const float *row = &bufB[c * T_LEN];
            for (int t = 0; t < T_LEN; t++) {
                sum += row[t];
            }
            gap[c] = sum / (float)T_LEN;
        }

        // 16. Classifier: Linear(c_c5_out -> 2)
        float logit_0 = dm->classifier->biases ? dm->classifier->biases[0] : 0.0f;
        float logit_1 = dm->classifier->biases ? dm->classifier->biases[1] : 0.0f;
        const float *w_clf = dm->classifier->weights;
        for (int c = 0; c < c_c5_out; c++) {
            logit_0 += gap[c] * w_clf[c];
            logit_1 += gap[c] * w_clf[c_c5_out + c];
        }

        float logit_diff = logit_1 - logit_0;
#if CONFIG_PULSEVAD_FAST_SIGMOID
        return 0.5f + 0.5f * (logit_diff / (1.0f + fabsf(logit_diff)));
#else
        return 1.0f / (1.0f + expf(-logit_diff));
#endif
    }
}
