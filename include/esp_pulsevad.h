/**
 * ESP-IDF PulseVAD - Ultra-Tiny 2.1k Voice Activity Detector for ESP32-P4
 * =======================================================================
 * Target: ESP32-P4 RISC-V (Hardware FPU & PIE)
 * Author: Strg-Alt-Entf-0x00
 * License: MIT
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PULSEVAD_CHUNK_SAMPLES       3200     // 200 ms @ 16 kHz
#define PULSEVAD_SAMPLE_RATE_HZ      16000    // Required audio sample rate
#define PULSEVAD_DEFAULT_THRESHOLD   0.50f    // Default speech probability threshold

/**
 * PulseVAD Configuration Parameters
 */
typedef struct {
    float speech_threshold;          // Probability threshold (0.0 - 1.0, default 0.50f)
    uint32_t min_speech_ms;          // Hysteresis: min speech duration to trigger active (default 100 ms)
    uint32_t min_silence_ms;         // Hysteresis: hangover silence duration before release (default 200 ms)
} pulsevad_config_t;

/**
 * Opaque PulseVAD Engine Handle
 */
typedef struct pulsevad_engine_s* pulsevad_handle_t;

/**
 * Get default recommended configuration
 */
static inline pulsevad_config_t pulsevad_default_config(void) {
    pulsevad_config_t cfg = {
        .speech_threshold = PULSEVAD_DEFAULT_THRESHOLD,
        .min_speech_ms = 100,
        .min_silence_ms = 200,
    };
    return cfg;
}

/**
 * Initialize PulseVAD engine instance
 *
 * @param config Pointer to configuration struct (or NULL for defaults)
 * @param out_handle Pointer to receive initialized engine handle
 * @return ESP_OK on success, ESP_ERR_NO_MEM if allocation fails
 */
esp_err_t pulsevad_init(const pulsevad_config_t *config, pulsevad_handle_t *out_handle);

/**
 * Process a 200 ms chunk of float32 audio samples (-1.0 to 1.0)
 *
 * @param handle Initialized engine handle
 * @param audio_3200 Pointer to exactly 3,200 contiguous float32 samples @ 16 kHz
 * @param out_prob Pointer to receive speech probability (0.0 to 1.0, can be NULL)
 * @param out_is_speech Pointer to receive boolean speech decision (after hysteresis, can be NULL)
 * @return ESP_OK on success
 */
esp_err_t pulsevad_process_chunk(pulsevad_handle_t handle,
                                const float *audio_3200,
                                float *out_prob,
                                bool *out_is_speech);

/**
 * Process a 200 ms chunk of 16-bit PCM audio samples (-32768 to 32767)
 * (Normalizes samples inline without allocating an intermediate float audio buffer)
 *
 * @param handle Initialized engine handle
 * @param pcm_3200 Pointer to exactly 3,200 contiguous int16_t samples @ 16 kHz
 * @param out_prob Pointer to receive speech probability (0.0 to 1.0, can be NULL)
 * @param out_is_speech Pointer to receive boolean speech decision (can be NULL)
 * @return ESP_OK on success
 */
esp_err_t pulsevad_process_i16_chunk(pulsevad_handle_t handle,
                                    const int16_t *pcm_3200,
                                    float *out_prob,
                                    bool *out_is_speech);

/**
 * Update the speech probability threshold used by the temporal state machine.
 *
 * @param handle Initialized engine handle
 * @param threshold Probability threshold in the inclusive range [0.0, 1.0]
 * @return ESP_OK on success, ESP_ERR_INVALID_ARG for an invalid handle or threshold
 */
esp_err_t pulsevad_set_threshold(pulsevad_handle_t handle, float threshold);

/**
 * Update minimum speech duration required to trigger active speech state.
 *
 * @param handle Initialized engine handle
 * @param min_speech_ms Minimum speech duration in ms (default: 100)
 * @return ESP_OK on success
 */
esp_err_t pulsevad_set_min_speech_ms(pulsevad_handle_t handle, uint32_t min_speech_ms);

/**
 * Update minimum silence duration before releasing speech state (hangover).
 *
 * @param handle Initialized engine handle
 * @param min_silence_ms Hangover silence duration in ms (default: 200)
 * @return ESP_OK on success
 */
esp_err_t pulsevad_set_min_silence_ms(pulsevad_handle_t handle, uint32_t min_silence_ms);

/**
 * Reset temporal hysteresis state (silence/speech duration counters)
 *
 * @param handle Initialized engine handle
 */
void pulsevad_reset(pulsevad_handle_t handle);

/**
 * Free PulseVAD engine instance and release allocated memory
 *
 * @param handle Engine handle to destroy
 */
void pulsevad_free(pulsevad_handle_t handle);

/**
 * Run bit-exact golden self-test against precomputed reference vectors
 *
 * Asserts that the ESP32-P4 DSP & FPU inference matches PyTorch within 1e-4 tolerance.
 *
 * @return ESP_OK if self-test passes, ESP_FAIL otherwise
 */
esp_err_t pulsevad_run_self_test(void);

/**
 * Load a dynamic PulseVAD model (.espm binary) from a filesystem path.
 *
 * Supports the PulseVAD student and teacher model architectures in ESPM v3.
 *
 * @param handle Engine instance
 * @param espm_path Path to an ESPM model file (e.g. "/sd/models/pulsevad_student.espm")
 * @return ESP_OK on success
 */
esp_err_t pulsevad_load_model_espm_file(pulsevad_handle_t handle, const char *espm_path);

/**
 * Reset model to the built-in ROM Student model
 *
 * @param handle Engine instance
 * @return ESP_OK on success
 */
esp_err_t pulsevad_reset_to_default_model(pulsevad_handle_t handle);

/**
 * Get human-readable description of the currently active model
 *
 * @param handle Engine instance
 * @param out_buf Buffer to receive description string
 * @param buf_len Buffer size in bytes
 * @return ESP_OK on success
 */
esp_err_t pulsevad_get_model_info(pulsevad_handle_t handle, char *out_buf, size_t buf_len);

#ifdef __cplusplus
}
#endif
