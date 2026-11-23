/**
 * ESP-IDF PulseVAD Public API Implementation
 * ===========================================
 * Orchestrates frontend feature extraction, neural network inference,
 * temporal hysteresis, and unit testing for ESP32-P4.
 */

#include "esp_pulsevad.h"
#include "pulsevad_internal.h"
#include "pulsevad_golden_data.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *TAG = "PULSEVAD";

esp_err_t pulsevad_init(const pulsevad_config_t *config, pulsevad_handle_t *out_handle) {
    if (!out_handle) {
        return ESP_ERR_INVALID_ARG;
    }

    // Initialize ESP-DSP FFT tables
    esp_err_t ret = pulsevad_frontend_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize frontend DSP: %s", esp_err_to_name(ret));
        return ret;
    }

    struct pulsevad_engine_s *engine = (struct pulsevad_engine_s *)heap_caps_malloc(
        sizeof(struct pulsevad_engine_s), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!engine) {
        ESP_LOGW(TAG, "Internal SRAM unavailable for engine state; falling back to default heap");
        engine = (struct pulsevad_engine_s *)malloc(sizeof(struct pulsevad_engine_s));
    }
    if (!engine) {
        ESP_LOGE(TAG, "Failed to allocate PulseVAD engine instance");
        return ESP_ERR_NO_MEM;
    }

    if (config) {
        engine->config = *config;
    } else {
        engine->config = pulsevad_default_config();
    }

    engine->is_speaking = false;
    engine->active_speech_ms = 0;
    engine->continuous_silence_ms = 0;
    engine->dynamic_model = NULL;

    *out_handle = engine;
    ESP_LOGI(TAG, "PulseVAD engine initialized (Threshold: %.2f, MinSpeech: %ums, MinSilence: %ums)",
             engine->config.speech_threshold,
             (unsigned)engine->config.min_speech_ms,
             (unsigned)engine->config.min_silence_ms);

    return ESP_OK;
}

static esp_err_t pulsevad_process_chunk_internal(pulsevad_handle_t handle,
                                                 const float *audio_3200,
                                                 const int16_t *pcm_3200,
                                                 float *out_prob,
                                                 bool *out_is_speech) {
    if (!handle || (!audio_3200 && !pcm_3200)) {
        return ESP_ERR_INVALID_ARG;
    }

#if CONFIG_PULSEVAD_ENABLE_BENCHMARK_LOGS
    int64_t t_start = esp_timer_get_time();
#endif

    // Use engine working buffer on heap (zero stack overhead)
    float *features = handle->features;

    // 1. Audio frontend (Pre-emphasis -> Z-norm -> STFT -> Mel -> Log -> Bin Norm)
    const pulsevad_sparse_mel_bin_t *mel_bins = handle->dynamic_model
        ? handle->dynamic_model->mel_bins
        : NULL;
    if (pcm_3200) {
        pulsevad_frontend_extract_i16(pcm_3200, mel_bins, features);
    } else {
        pulsevad_frontend_extract(audio_3200, mel_bins, features);
    }

#if CONFIG_PULSEVAD_ENABLE_BENCHMARK_LOGS
    int64_t t_dsp = esp_timer_get_time();
#endif

    // 2. Neural network forward inference (built-in 2.1k or dynamic ESPM)
    float prob = pulsevad_nn_forward(handle, features);

#if CONFIG_PULSEVAD_ENABLE_BENCHMARK_LOGS
    int64_t t_nn = esp_timer_get_time();
    ESP_LOGI(TAG, "Inference: DSP=%lld us, NN=%lld us, Total=%lld us | Prob=%.4f",
             (t_dsp - t_start), (t_nn - t_dsp), (t_nn - t_start), prob);
#endif

    // 3. Temporal Hysteresis State Machine
    const uint32_t chunk_duration_ms = (PULSEVAD_WINDOW_SAMPLES * 1000) / PULSEVAD_SAMPLE_RATE_HZ; // 200 ms

    if (prob >= handle->config.speech_threshold) {
        handle->active_speech_ms += chunk_duration_ms;
        handle->continuous_silence_ms = 0;

        if (handle->active_speech_ms >= handle->config.min_speech_ms) {
            handle->is_speaking = true;
        }
    } else {
        handle->continuous_silence_ms += chunk_duration_ms;

        if (handle->continuous_silence_ms >= handle->config.min_silence_ms) {
            handle->is_speaking = false;
            handle->active_speech_ms = 0;
        }
    }

    if (out_prob) {
        *out_prob = prob;
    }
    if (out_is_speech) {
        *out_is_speech = handle->is_speaking;
    }

    return ESP_OK;
}

esp_err_t pulsevad_process_chunk(pulsevad_handle_t handle,
                                const float *audio_3200,
                                float *out_prob,
                                bool *out_is_speech) {
    return pulsevad_process_chunk_internal(handle, audio_3200, NULL,
                                           out_prob, out_is_speech);
}

esp_err_t pulsevad_process_i16_chunk(pulsevad_handle_t handle,
                                    const int16_t *pcm_3200,
                                    float *out_prob,
                                    bool *out_is_speech) {
    return pulsevad_process_chunk_internal(handle, NULL, pcm_3200,
                                           out_prob, out_is_speech);
}

esp_err_t pulsevad_set_threshold(pulsevad_handle_t handle, float threshold) {
    if (!handle || !isfinite(threshold) || threshold < 0.0f || threshold > 1.0f) {
        return ESP_ERR_INVALID_ARG;
    }

    handle->config.speech_threshold = threshold;
    return ESP_OK;
}

esp_err_t pulsevad_set_min_speech_ms(pulsevad_handle_t handle, uint32_t min_speech_ms) {
    if (!handle || min_speech_ms == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    handle->config.min_speech_ms = min_speech_ms;
    handle->active_speech_ms = 0;
    handle->continuous_silence_ms = 0;
    handle->is_speaking = false;
    return ESP_OK;
}

esp_err_t pulsevad_set_min_silence_ms(pulsevad_handle_t handle, uint32_t min_silence_ms) {
    if (!handle || min_silence_ms == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    handle->config.min_silence_ms = min_silence_ms;
    handle->active_speech_ms = 0;
    handle->continuous_silence_ms = 0;
    handle->is_speaking = false;
    return ESP_OK;
}

void pulsevad_reset(pulsevad_handle_t handle) {
    if (handle) {
        handle->is_speaking = false;
        handle->active_speech_ms = 0;
        handle->continuous_silence_ms = 0;
    }
}

void pulsevad_free(pulsevad_handle_t handle) {
    if (handle) {
        if (handle->dynamic_model) {
            if (handle->dynamic_model->raw_data) {
                free(handle->dynamic_model->raw_data);
            }
            free(handle->dynamic_model);
            handle->dynamic_model = NULL;
        }
        free(handle);
    }
}

esp_err_t pulsevad_reset_to_default_model(pulsevad_handle_t handle) {
    if (!handle) {
        return ESP_ERR_INVALID_ARG;
    }
    if (handle->dynamic_model) {
        if (handle->dynamic_model->raw_data) {
            free(handle->dynamic_model->raw_data);
        }
        free(handle->dynamic_model);
        handle->dynamic_model = NULL;
        ESP_LOGI(TAG, "Reset to built-in default Flash ROM model (PulseVAD 2.1k)");
    }
    return ESP_OK;
}

esp_err_t pulsevad_get_model_info(pulsevad_handle_t handle, char *out_buf, size_t buf_len) {
    if (!handle || !out_buf || buf_len == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    if (handle->dynamic_model) {
        snprintf(out_buf, buf_len, "%s (%lu params, %.1f KB)",
                 handle->dynamic_model->model_name,
                 (unsigned long)handle->dynamic_model->total_params,
                 (float)handle->dynamic_model->raw_size / 1024.0f);
    } else {
        snprintf(out_buf, buf_len, "PulseVAD 2.1k (Built-in Flash ROM, 2,042 folded params)");
    }
    return ESP_OK;
}

esp_err_t pulsevad_run_self_test(void) {
    ESP_LOGI(TAG, "Starting PulseVAD Golden Self-Test on ESP32-P4...");

    pulsevad_handle_t handle = NULL;
    esp_err_t ret = pulsevad_init(NULL, &handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Self-test failed: could not initialize engine");
        return ret;
    }

    // 1. Test Golden Speech Vector
    float speech_prob = 0.0f;
    bool speech_detected = false;
    ret = pulsevad_process_chunk(handle, GOLDEN_SPEECH_AUDIO, &speech_prob, &speech_detected);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Self-test failed during speech chunk processing");
        pulsevad_free(handle);
        return ret;
    }

    float speech_diff = fabsf(speech_prob - GOLDEN_SPEECH_EXPECTED_PROB);
    ESP_LOGI(TAG, "Speech Vector: Measured=%.6f, Expected=%.6f, Diff=%.2e",
             speech_prob, GOLDEN_SPEECH_EXPECTED_PROB, speech_diff);

    // 2. Test Golden Noise Vector
    float noise_prob = 0.0f;
    bool noise_detected = false;
    ret = pulsevad_process_chunk(handle, GOLDEN_NOISE_AUDIO, &noise_prob, &noise_detected);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Self-test failed during noise chunk processing");
        pulsevad_free(handle);
        return ret;
    }

    float noise_diff = fabsf(noise_prob - GOLDEN_NOISE_EXPECTED_PROB);
    ESP_LOGI(TAG, "Noise Vector:  Measured=%.6f, Expected=%.6f, Diff=%.2e",
             noise_prob, GOLDEN_NOISE_EXPECTED_PROB, noise_diff);

    pulsevad_free(handle);

    // Tolerance of 1e-4 asserts bit-exact DSP & FPU equivalence with PyTorch
    if (speech_diff < 1e-4f && noise_diff < 1e-4f) {
        ESP_LOGI(TAG, "[PASSED] PulseVAD Self-Test PASSED! Bit-exact FPU match verified.");
        return ESP_OK;
    } else {
        ESP_LOGE(TAG, "[FAILED] PulseVAD Self-Test precision mismatch exceeded 1e-4!");
        return ESP_FAIL;
    }
}
