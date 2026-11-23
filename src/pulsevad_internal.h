/**
 * PulseVAD Internal Definitions & Engine State
 * ============================================
 * Internal header for PulseVAD implementation.
 */

#pragma once

#include "esp_pulsevad.h"
#include "pulsevad_model_data.h"
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Internal Layer Descriptor for dynamic models
 */
typedef struct {
    char name[32];
    uint16_t w_shape[3];
    uint16_t has_bias;
    uint32_t num_weights;
    uint32_t num_biases;
    float *weights;
    float *biases;
} pulsevad_layer_t;

typedef struct {
    char model_name[32];
    uint32_t total_params;
    float prior_bias_shift;
    uint8_t *raw_data;
    size_t raw_size;

    pulsevad_sparse_mel_bin_t mel_bins[PULSEVAD_NUM_MELS];
    pulsevad_layer_t layers[14];

    // Direct pointers to key layers
    pulsevad_layer_t *adapter;
    pulsevad_layer_t *conv0_dw;
    pulsevad_layer_t *conv0_pw;
    pulsevad_layer_t *block1;
    pulsevad_layer_t *block2;
    pulsevad_layer_t *subA_dw;
    pulsevad_layer_t *subA_pw;
    pulsevad_layer_t *subC_dw;
    pulsevad_layer_t *subC_pw;
    pulsevad_layer_t *skip;
    pulsevad_layer_t *conv4_dw;
    pulsevad_layer_t *conv4_pw;
    pulsevad_layer_t *conv5;
    pulsevad_layer_t *classifier;
} pulsevad_dynamic_model_t;

/**
 * Internal PulseVAD Engine State
 */
struct pulsevad_engine_s {
    pulsevad_config_t config;
    
    // Hysteresis temporal state
    bool is_speaking;
    uint32_t active_speech_ms;
    uint32_t continuous_silence_ms;

    // Optional dynamic ESPM model (NULL = use built-in 2.1k Flash ROM)
    pulsevad_dynamic_model_t *dynamic_model;

    // Heap-allocated working buffers (avoids FreeRTOS task stack overflows)
    float features[PULSEVAD_NUM_MELS * PULSEVAD_NUM_FRAMES]; // 1,344 floats (~5.3 KB)

    // Intermediate CNN layer buffers (sized for max 128 channels x 21 frames = 2688 floats = ~10.7 KB)
    float nn_bufA[128 * PULSEVAD_NUM_FRAMES];
    float nn_bufB[128 * PULSEVAD_NUM_FRAMES];
    float nn_res[64 * PULSEVAD_NUM_FRAMES];
    float nn_main[64 * PULSEVAD_NUM_FRAMES];
};

/**
 * Initialize ESP-DSP FFT tables for PulseVAD (512-pt FFT)
 */
esp_err_t pulsevad_frontend_init(void);

/**
 * Extract 64-channel x 21-frame normalized log-Mel spectrogram
 *
 * @param audio_3200 Contiguous 3,200 audio samples (16 kHz mono)
 * @param out_features_64x21 Output buffer for 64x21 floats (1,344 floats)
 */
void pulsevad_frontend_extract(const float *audio_3200,
                               const pulsevad_sparse_mel_bin_t *mel_bins,
                               float *out_features_64x21);
void pulsevad_frontend_extract_i16(const int16_t *audio_3200,
                                   const pulsevad_sparse_mel_bin_t *mel_bins,
                                   float *out_features_64x21);

/**
 * Execute forward inference of the FoldedPulseVAD neural network
 *
 * @param handle Engine instance
 * @param features_64x21 Pointer to 64x21 extracted log-mel features
 * @return Speech probability float in [0.0, 1.0]
 */
float pulsevad_nn_forward(pulsevad_handle_t handle, const float *features_64x21);

#ifdef __cplusplus
}
#endif
