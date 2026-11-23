/**
 * PulseVAD Audio Frontend Implementation
 * ====================================================
 * Computes 64-channel log-Mel spectrogram over 200 ms causal audio window.
 * Uses the ESP-DSP 512-point FFT; the selected implementation depends on the
 * configured target and ESP-DSP build.
 *
 * NOTE: Working buffers are placed in BSS/SRAM (not task stack) to prevent
 * FreeRTOS task stack overflows in constrained environments.
 */

#include "pulsevad_internal.h"
#include "dsps_fft4r.h"
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Static precomputed Hann window (400 samples periodic)
static float s_hann_window[PULSEVAD_WIN_LENGTH];
static bool s_frontend_initialized = false;

// Real FFT input: 512 samples packed as 256 complex values.
static float s_fft_buf[PULSEVAD_N_FFT] __attribute__((aligned(16)));

// Intermediate DSP working buffers in BSS (zero task stack overhead)
static float s_x_pre[PULSEVAD_WINDOW_SAMPLES];
static float s_x_padded[PULSEVAD_WINDOW_SAMPLES + PULSEVAD_N_FFT]; // 3,712 floats
static float s_log_mel[PULSEVAD_NUM_MELS][PULSEVAD_NUM_FRAMES];   // 64x21 floats
static float s_power_spectrum[PULSEVAD_N_FFT / 2 + 1];             // 257 floats

esp_err_t pulsevad_frontend_init(void) {
    if (s_frontend_initialized) {
        return ESP_OK;
    }

    // Pack the real 512-point STFT into a 256-point complex radix-4 FFT.
    esp_err_t ret = dsps_fft4r_init_fc32(NULL, PULSEVAD_N_FFT / 2);
    if (ret != ESP_OK) {
        return ret;
    }

    // 2. Precompute periodic Hann window: 0.5 - 0.5 * cos(2 * pi * n / 400)
    for (int n = 0; n < PULSEVAD_WIN_LENGTH; n++) {
        s_hann_window[n] = 0.5f - 0.5f * cosf((2.0f * (float)M_PI * (float)n) / (float)PULSEVAD_WIN_LENGTH);
    }

    s_frontend_initialized = true;
    return ESP_OK;
}

static void pulsevad_frontend_extract_preemphasized(
    const pulsevad_sparse_mel_bin_t *mel_bins,
    float *out_features_64x21,
    float sum
) {
    if (!mel_bins) {
        mel_bins = PULSEVAD_MEL_BINS;
    }

    // 2. Waveform z-norm: (x - mean) / (std + EPS) with ddof=0
    float mean = sum / (float)PULSEVAD_WINDOW_SAMPLES;

    float sum_sq = 0.0f;
    for (int i = 0; i < PULSEVAD_WINDOW_SAMPLES; i++) {
        float diff = s_x_pre[i] - mean;
        sum_sq += diff * diff;
    }
    float std = sqrtf(sum_sq / (float)PULSEVAD_WINDOW_SAMPLES);
    float inv_std = 1.0f / (std + PULSEVAD_EPS);

    // 3. Centered STFT: reflect pad 256 on each side (total length: 3,712)
    // Left reflect: index 1 maps to 255, index 2 maps to 254...
    // Center: indices 256..3455
    // Right reflect: index 3198 maps to 3456, index 3197 maps to 3457...
    const int pad = PULSEVAD_N_FFT / 2; // 256

    for (int i = 0; i < PULSEVAD_WINDOW_SAMPLES; i++) {
        s_x_padded[pad + i] = (s_x_pre[i] - mean) * inv_std;
    }

    for (int i = 0; i < pad; i++) {
        s_x_padded[pad - 1 - i] = s_x_padded[pad + 1 + i];
    }
    for (int i = 0; i < pad; i++) {
        s_x_padded[pad + PULSEVAD_WINDOW_SAMPLES + i] =
            s_x_padded[pad + PULSEVAD_WINDOW_SAMPLES - 2 - i];
    }

    // 4. Extract 21 frames
    for (int frame = 0; frame < PULSEVAD_NUM_FRAMES; frame++) {
        int start = frame * PULSEVAD_HOP_LENGTH;

        memset(s_fft_buf, 0, PULSEVAD_N_FFT * sizeof(float));

        // Center window of length 400 with 56 zeros on each side.
        // The real samples are packed in adjacent real/imaginary FFT slots.
        for (int k = 0; k < PULSEVAD_WIN_LENGTH; k++) {
            s_fft_buf[PULSEVAD_PAD_WIN + k] =
                s_x_padded[start + PULSEVAD_PAD_WIN + k] * s_hann_window[k];
        }

        dsps_fft4r_fc32(s_fft_buf, PULSEVAD_N_FFT / 2);
        dsps_bit_rev4r_fc32(s_fft_buf, PULSEVAD_N_FFT / 2);
        dsps_cplx2real_fc32(s_fft_buf, PULSEVAD_N_FFT / 2);

        // The real-FFT conversion packs DC and Nyquist in the first complex value.
        s_power_spectrum[0] = s_fft_buf[0] * s_fft_buf[0];
        s_power_spectrum[PULSEVAD_N_FFT / 2] = s_fft_buf[1] * s_fft_buf[1];
        for (int k = 1; k < PULSEVAD_N_FFT / 2; k++) {
            float re = s_fft_buf[2 * k];
            float im = s_fft_buf[2 * k + 1];
            s_power_spectrum[k] = re * re + im * im;
        }

        // Apply Sparse Mel Filterbank (500 multiplications instead of 16,448!)
        for (int m = 0; m < PULSEVAD_NUM_MELS; m++) {
            const pulsevad_sparse_mel_bin_t *bin = &mel_bins[m];
            float acc = 0.0f;
            const uint16_t start_k = bin->start_k;
            const uint16_t num_k = bin->num_k;
            const float *w = bin->weights;

            for (uint16_t k = 0; k < num_k; k++) {
                acc += s_power_spectrum[start_k + k] * w[k];
            }

            // Log mel with epsilon
            s_log_mel[m][frame] = logf(acc + PULSEVAD_EPS);
        }
    }

    // 5. Per-bin z-norm across the 21 time frames
    for (int m = 0; m < PULSEVAD_NUM_MELS; m++) {
        float b_sum = 0.0f;
        for (int t = 0; t < PULSEVAD_NUM_FRAMES; t++) {
            b_sum += s_log_mel[m][t];
        }
        float b_mean = b_sum / (float)PULSEVAD_NUM_FRAMES;

        float b_sq = 0.0f;
        for (int t = 0; t < PULSEVAD_NUM_FRAMES; t++) {
            float diff = s_log_mel[m][t] - b_mean;
            b_sq += diff * diff;
        }
        float b_std = sqrtf(b_sq / (float)PULSEVAD_NUM_FRAMES);
        float inv_b_std = 1.0f / (b_std + PULSEVAD_EPS);

        for (int t = 0; t < PULSEVAD_NUM_FRAMES; t++) {
            out_features_64x21[m * PULSEVAD_NUM_FRAMES + t] = (s_log_mel[m][t] - b_mean) * inv_b_std;
        }
    }
}

void pulsevad_frontend_extract(const float *audio_3200,
                               const pulsevad_sparse_mel_bin_t *mel_bins,
                               float *out_features_64x21) {
    // 1. Pre-emphasis: y[n] = x[n] - 0.97 * x[n-1], y[0] = x[0]
    s_x_pre[0] = audio_3200[0];
    float sum = s_x_pre[0];
    for (int i = 1; i < PULSEVAD_WINDOW_SAMPLES; i++) {
        s_x_pre[i] = audio_3200[i] - PULSEVAD_PREEMPHASIS_ALPHA * audio_3200[i - 1];
        sum += s_x_pre[i];
    }

    pulsevad_frontend_extract_preemphasized(mel_bins, out_features_64x21, sum);
}

void pulsevad_frontend_extract_i16(const int16_t *audio_3200,
                                   const pulsevad_sparse_mel_bin_t *mel_bins,
                                   float *out_features_64x21) {
    float previous = (float)audio_3200[0] / 32768.0f;
    s_x_pre[0] = previous;
    float sum = previous;
    for (int i = 1; i < PULSEVAD_WINDOW_SAMPLES; i++) {
        const float current = (float)audio_3200[i] / 32768.0f;
        s_x_pre[i] = current - PULSEVAD_PREEMPHASIS_ALPHA * previous;
        sum += s_x_pre[i];
        previous = current;
    }

    pulsevad_frontend_extract_preemphasized(mel_bins, out_features_64x21, sum);
}
