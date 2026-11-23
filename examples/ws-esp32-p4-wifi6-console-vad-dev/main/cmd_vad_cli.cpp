/**
 * @file cmd_vad_cli.cpp
 * @brief PulseVAD Console Commands Implementation
 * 
 * Complete command-line interface exposing all PulseVAD parameters:
 * - Model loading (student / teacher from SD)
 * - VAD configuration (threshold, hysteresis)
 * - Microphone control (ES8311 / INMP441)
 * - WAV file testing
 * - Audio playback & recording
 * - System information
 */

#include "cmd_vad_cli.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include "esp_log.h"
#include "esp_console.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_pm.h"
#include "esp_private/esp_clk.h"
#include "argtable3/argtable3.h"

#include "app_config.h"
#include "audio_manager.h"
#include "board_audio_config.h"
#include "audio_wav.h"
#include "esp_pulsevad.h"

static const char* TAG = "VAD_CLI";

class InferenceCpuFrequencyLock {
public:
    esp_err_t acquire()
    {
#if CONFIG_PM_ENABLE
        esp_err_t result = esp_pm_lock_create(
            ESP_PM_CPU_FREQ_MAX, 0, "vad_test", &handle_);
        if (result != ESP_OK) {
            return result;
        }

        result = esp_pm_lock_acquire(handle_);
        if (result != ESP_OK) {
            const esp_err_t delete_result = esp_pm_lock_delete(handle_);
            handle_ = nullptr;
            if (delete_result != ESP_OK) {
                ESP_LOGE(TAG, "Failed to delete CPU frequency lock: %s",
                         esp_err_to_name(delete_result));
            }
        }
        return result;
#else
        return ESP_OK;
#endif
    }

    ~InferenceCpuFrequencyLock()
    {
#if CONFIG_PM_ENABLE
        if (handle_) {
            const esp_err_t release_result = esp_pm_lock_release(handle_);
            if (release_result != ESP_OK) {
                ESP_LOGE(TAG, "Failed to release CPU frequency lock: %s",
                         esp_err_to_name(release_result));
            }
            const esp_err_t delete_result = esp_pm_lock_delete(handle_);
            if (delete_result != ESP_OK) {
                ESP_LOGE(TAG, "Failed to delete CPU frequency lock: %s",
                         esp_err_to_name(delete_result));
            }
        }
#endif
    }

private:
#if CONFIG_PM_ENABLE
    esp_pm_lock_handle_t handle_ = nullptr;
#endif
};

// ============================================================================
// Global State
// ============================================================================

static pulsevad_handle_t g_vad_handle = NULL;
static bool g_model_loaded = false;
static char g_model_path[MAX_FILE_PATH] = {0};
static float g_threshold = DEFAULT_VAD_THRESHOLD;

static void register_command(const char *command,
                             const char *help,
                             esp_console_cmd_func_t func,
                             void *argtable = NULL) {
    esp_console_cmd_t definition = {};
    definition.command = command;
    definition.help = help;
    definition.func = func;
    definition.argtable = argtable;
    ESP_ERROR_CHECK(esp_console_cmd_register(&definition));
}

static esp_err_t ensure_vad_initialized()
{
    if (g_vad_handle) {
        return ESP_OK;
    }
    pulsevad_config_t config = pulsevad_default_config();
    config.speech_threshold = g_threshold;
    return pulsevad_init(&config, &g_vad_handle);
}

// ============================================================================
// Helper: Write WAV Header
// ============================================================================

static esp_err_t write_wav_header(FILE* file, uint32_t data_size)
{
    return audio_wav_write_pcm16_header(
        file, PULSEVAD_SAMPLE_RATE, PULSEVAD_NUM_CHANNELS, data_size);
}

// ============================================================================
// Helper: List Directory Contents
// ============================================================================

static void list_directory(const char* path, int level) {
    if (level > 3) return; // Limit recursion
    
    DIR* dir = opendir(path);
    if (!dir) {
        if (level == 0) printf("Failed to open directory: %s\n", path);
        return;
    }
    
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        char full_path[MAX_FILE_PATH];
        size_t path_length = strlen(path);
        size_t name_length = strlen(ent->d_name);
        if (path_length >= sizeof(full_path) ||
            name_length >= sizeof(full_path) - path_length - 1) {
            printf("Skipping overlong path entry: %s\n", ent->d_name);
            continue;
        }
        memcpy(full_path, path, path_length);
        full_path[path_length] = '/';
        memcpy(full_path + path_length + 1, ent->d_name, name_length + 1);
        
        struct stat st;
        if (stat(full_path, &st) != 0) continue;
        
        // Print indentation
        for (int i = 0; i < level; i++) printf("  ");
        
        if (S_ISDIR(st.st_mode)) {
            if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0) {
                printf("[DIR]  %s/\n", ent->d_name);
                list_directory(full_path, level + 1);
            }
        } else {
            printf("[FILE] %s (%ld bytes)\n", ent->d_name, st.st_size);
        }
    }
    closedir(dir);
}

// ============================================================================
// Command: info
// ============================================================================

static int cmd_info(int argc, char **argv) {
    printf("\n=== System Information ===\n");
    
    // Memory
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t free_heap = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    printf("PSRAM Free    : %zu KB\n", free_psram / 1024);
    printf("Heap Free     : %zu KB\n", free_heap / 1024);
    
    // Audio
    printf("\nAudio Manager : %s\n", audio_manager_is_initialized() ? "READY" : "NOT INITIALIZED");
    if (audio_manager_is_initialized()) {
        float gain = 0.0f;
        audio_manager_get_mic_gain(&gain);
        mic_type_t mic = audio_manager_get_mic();
        printf("Active Mic    : %s\n", mic == MIC_I2S1_INMP441 ? "INMP441 (I2S1)" : "ES8311 (I2S0)");
        printf("Mic Gain      : %.1f dB\n", gain);
        printf("Capture       : %s\n", audio_manager_is_capturing() ? "ACTIVE" : "IDLE");
        printf("Playback      : %s\n", audio_manager_is_playing() ? "ACTIVE" : "IDLE");
    }
    
    // VAD Model
    char model_info[128] = {};
    if (g_vad_handle && pulsevad_get_model_info(g_vad_handle, model_info, sizeof(model_info)) == ESP_OK) {
        printf("\nVAD Model     : %s\n", model_info);
    } else {
        printf("\nVAD Model     : not initialized\n");
    }
    if (g_model_loaded) {
        printf("Model Path    : %s\n", g_model_path);
    }
    printf("Threshold     : %.2f\n", g_threshold);
    
    printf("==========================\n\n");
    return 0;
}

// ============================================================================
// Command: model
// ============================================================================

static int cmd_model(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const esp_err_t init_result = ensure_vad_initialized();
    if (init_result != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize PulseVAD: %s", esp_err_to_name(init_result));
        return 1;
    }

    char model_info[128] = {};
    const esp_err_t info_result =
        pulsevad_get_model_info(g_vad_handle, model_info, sizeof(model_info));
    if (info_result != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get active model information: %s",
                 esp_err_to_name(info_result));
        return 1;
    }

    printf("VAD Model: %s\n", model_info);
    if (g_model_loaded) {
        printf("Model Path: %s\n", g_model_path);
    }
    return 0;
}

// ============================================================================
// Command: ls [path]
// ============================================================================

static struct {
    struct arg_str *path;
    struct arg_end *end;
} ls_args;

static int cmd_ls(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void**)&ls_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, ls_args.end, argv[0]);
        return 1;
    }
    
    const char* path = (ls_args.path->count > 0) ? ls_args.path->sval[0] : "/sd/models";
    
    printf("\nListing: %s\n", path);
    printf("=================================\n");
    list_directory(path, 0);
    printf("=================================\n\n");
    
    return 0;
}

// ============================================================================
// Command: load_model <path.espm>
// ============================================================================

static struct {
    struct arg_str *path;
    struct arg_end *end;
} load_model_args;

static int cmd_load_model(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void**)&load_model_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, load_model_args.end, argv[0]);
        return 1;
    }
    
    const char* model_path = load_model_args.path->sval[0];
    
    ESP_LOGI(TAG, "Loading model: %s", model_path);
    const bool created_handle = g_vad_handle == NULL;
    const esp_err_t init_result = ensure_vad_initialized();
    if (init_result != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize PulseVAD: %s", esp_err_to_name(init_result));
        return 1;
    }

    const esp_err_t ret = pulsevad_load_model_espm_file(g_vad_handle, model_path);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load model: %s", esp_err_to_name(ret));
        if (created_handle) {
            pulsevad_free(g_vad_handle);
            g_vad_handle = NULL;
        }
        return 1;
    }
    
    g_model_loaded = true;
    snprintf(g_model_path, sizeof(g_model_path), "%s", model_path);
    
    printf("[OK] Model loaded successfully\n");
    printf("     Path: %s\n", model_path);
    printf("     Threshold: %.2f\n", g_threshold);
    
    return 0;
}

// ============================================================================
// Command: unload_model
// ============================================================================

static int cmd_unload_model(int argc, char **argv) {
    if (!g_model_loaded) {
        if (ensure_vad_initialized() != ESP_OK) {
            printf("[ERROR] PulseVAD engine initialization failed\n");
            return 1;
        }
        printf("[INFO] Using the built-in ROM model\n");
        return 0;
    }
    
    const esp_err_t ret = pulsevad_reset_to_default_model(g_vad_handle);
    if (ret != ESP_OK) {
        printf("[ERROR] Could not reset to built-in ROM model: %s\n", esp_err_to_name(ret));
        return 1;
    }
    g_model_loaded = false;
    memset(g_model_path, 0, sizeof(g_model_path));
    
    printf("[OK] Reset to built-in ROM model\n");
    return 0;
}

// ============================================================================
// Command: threshold <0.0-1.0>
// ============================================================================

static struct {
    struct arg_dbl *value;
    struct arg_end *end;
} threshold_args;

static int cmd_threshold(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void**)&threshold_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, threshold_args.end, argv[0]);
        return 1;
    }
    
    float new_threshold = (float)threshold_args.value->dval[0];
    
    if (!isfinite(new_threshold) ||
        new_threshold < MIN_VAD_THRESHOLD ||
        new_threshold > MAX_VAD_THRESHOLD) {
        printf("[ERROR] Threshold must be between %.1f and %.1f\n", 
               MIN_VAD_THRESHOLD, MAX_VAD_THRESHOLD);
        return 1;
    }

    // Update the active engine when it is already initialized.
    if (g_vad_handle) {
        esp_err_t ret = pulsevad_set_threshold(g_vad_handle, new_threshold);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to update PulseVAD threshold: %s", esp_err_to_name(ret));
            return 1;
        }
    }

    g_threshold = new_threshold;
    printf("[OK] Speech threshold set to %.2f\n", g_threshold);
    return 0;
}

// ============================================================================
// Command: mic status | mic select <onboard|i2s> | mic gain <dB>
// ============================================================================

static int cmd_mic(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "status") == 0) {
        if (!audio_manager_is_initialized()) {
            printf("[ERROR] Audio manager is not initialized.\n");
            return 1;
        }

        float gain_db = 0.0f;
        esp_err_t ret = audio_manager_get_mic_gain(&gain_db);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Could not read microphone gain: %s", esp_err_to_name(ret));
            return 1;
        }

        const mic_type_t mic = audio_manager_get_mic();
        printf("Input: %s\nCapture: %s\nSample format: 16000 Hz, mono, signed PCM16\n"
               "Onboard ES8311 PGA gain: %.0f dB\n"
               "External INMP441 input: optional\n",
               mic == MIC_I2S1_INMP441 ? "external INMP441" : "onboard ES8311",
               audio_manager_is_capturing() ? "running" : "stopped",
               gain_db);
        return 0;
    }

    if (argc == 3 && strcmp(argv[1], "select") == 0) {
        mic_type_t mic;
        if (strcmp(argv[2], "onboard") == 0) {
            mic = MIC_I2S0_ES8311;
        } else if (strcmp(argv[2], "i2s") == 0) {
            mic = MIC_I2S1_INMP441;
        } else {
            printf("Usage: mic select <onboard|i2s>\n");
            return 1;
        }

        const esp_err_t ret = audio_manager_set_mic(mic);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Could not select microphone: %s", esp_err_to_name(ret));
            return 1;
        }
        printf("[OK] Selected %s (saved across restarts)\n",
               mic == MIC_I2S1_INMP441 ? "external INMP441" : "onboard ES8311");
        return 0;
    }

    if (argc == 3 && strcmp(argv[1], "gain") == 0) {
        char *end = NULL;
        const float gain_db = strtof(argv[2], &end);
        if (end == argv[2] || *end != '\0') {
            printf("Gain must be one of 0, 6, 12, 18, 24, 30, 36, or 42 dB.\n");
            return 1;
        }

        const esp_err_t ret = audio_manager_set_mic_gain(gain_db);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Could not set microphone gain: %s", esp_err_to_name(ret));
            return 1;
        }
        printf("[OK] Onboard microphone gain set to %.0f dB (saved across restarts)\n",
               gain_db);
        return 0;
    }

    printf("Usage: mic status | mic select <onboard|i2s> | "
           "mic gain <0|6|12|18|24|30|36|42>\n");
    return 1;
}

// ============================================================================
// Command: speaker_vol <0-100>
// ============================================================================

static struct {
    struct arg_int *volume;
    struct arg_end *end;
} speaker_vol_args;

static int cmd_speaker_vol(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void**)&speaker_vol_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, speaker_vol_args.end, argv[0]);
        return 1;
    }
    
    int vol = speaker_vol_args.volume->ival[0];
    
    if (vol < 0 || vol > 100) {
        printf("[ERROR] Volume must be between 0 and 100\n");
        return 1;
    }
    
    esp_err_t ret = audio_manager_set_speaker_vol((uint8_t)vol);
    if (ret == ESP_OK) {
        printf("[OK] Speaker volume set to %d%% (saved across restarts)\n", vol);
    } else {
        printf("[ERROR] Failed to apply or save speaker volume: %s\n",
               esp_err_to_name(ret));
        return 1;
    }
    
    return 0;
}

// ============================================================================
// Command: test_wav <filename.wav>
// ============================================================================

static struct {
    struct arg_str *file;
    struct arg_end *end;
} test_wav_args;

static int cmd_test_wav(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void**)&test_wav_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, test_wav_args.end, argv[0]);
        return 1;
    }
    
    if (ensure_vad_initialized() != ESP_OK) {
        printf("[ERROR] PulseVAD engine initialization failed\n");
        return 1;
    }
    
    const char* wav_path = test_wav_args.file->sval[0];
    
    ESP_LOGI(TAG, "Testing WAV file: %s", wav_path);
    
    audio_wav_handle_t wav = NULL;
    esp_err_t ret = audio_wav_open(wav_path, &wav);
    if (ret != ESP_OK) {
        printf("[ERROR] Failed to open WAV file: %s (%s)\n",
               wav_path, esp_err_to_name(ret));
        return 1;
    }
    
    audio_wav_info_t wav_info {};
    ret = audio_wav_get_info(wav, &wav_info);
    if (ret != ESP_OK) {
        printf("[ERROR] Failed to read WAV metadata: %s\n", esp_err_to_name(ret));
        audio_wav_close(wav);
        return 1;
    }
    
    if (wav_info.sample_rate != PULSEVAD_SAMPLE_RATE ||
        wav_info.num_channels != PULSEVAD_NUM_CHANNELS ||
        wav_info.bits_per_sample != PULSEVAD_BITS_PER_SAMPLE ||
        wav_info.data_size % sizeof(int16_t) != 0) {
        printf("[ERROR] VAD requires %d Hz mono PCM16 with complete samples; got %" PRIu32
               " Hz, %u channel(s), %u-bit, "
               "%" PRIu32 " bytes\n",
               PULSEVAD_SAMPLE_RATE, wav_info.sample_rate,
               (unsigned)wav_info.num_channels,
               (unsigned)wav_info.bits_per_sample, wav_info.data_size);
        audio_wav_close(wav);
        return 1;
    }

    pulsevad_reset(g_vad_handle);
    InferenceCpuFrequencyLock cpu_lock;
    const esp_err_t lock_error = cpu_lock.acquire();
    if (lock_error != ESP_OK) {
        printf("[ERROR] Failed to lock CPU frequency for measurement: %s\n",
               esp_err_to_name(lock_error));
        audio_wav_close(wav);
        return 1;
    }
    printf("[INFO] CPU frequency locked at %u MHz\n",
           (unsigned)(esp_clk_cpu_freq() / 1000000));
    
    // Process audio in 200ms chunks
    int16_t *chunk = (int16_t *)malloc(PULSEVAD_CHUNK_BYTES);
    if (!chunk) {
        printf("[ERROR] Failed to allocate WAV processing buffer\n");
        audio_wav_close(wav);
        return 1;
    }

    int chunk_num = 0;
    int64_t total_time_us = 0;
    uint32_t remaining_bytes = wav_info.data_size;
    const size_t chunk_bytes = PULSEVAD_CHUNK_BYTES;
    
    printf("\nProcessing...\n");
    printf("Chunk | Probability | Decision | Latency\n");
    printf("------+--------------+----------+---------\n");
    
    while (remaining_bytes >= chunk_bytes) {
        size_t bytes_read = 0;
        ret = audio_wav_read(wav, chunk, chunk_bytes, &bytes_read);
        if (ret != ESP_OK || bytes_read != chunk_bytes) {
            printf("[ERROR] Failed to read WAV audio data: %s\n",
                   ret == ESP_OK ? "unexpected end of audio" : esp_err_to_name(ret));
            free(chunk);
            audio_wav_close(wav);
            return 1;
        }
        remaining_bytes -= chunk_bytes;

        float prob = 0.0f;
        bool is_speech = false;
        
        int64_t start_us = esp_timer_get_time();
        ret = pulsevad_process_i16_chunk(g_vad_handle, chunk, &prob, &is_speech);
        int64_t end_us = esp_timer_get_time();
        if (ret != ESP_OK) {
            printf("[ERROR] PulseVAD inference failed: %s\n", esp_err_to_name(ret));
            free(chunk);
            audio_wav_close(wav);
            return 1;
        }
        
        int64_t latency_us = end_us - start_us;
        total_time_us += latency_us;
        
        printf("%5d | %11.6f | %8s | %6lld us\n",
               chunk_num++, prob, is_speech ? "SPEECH" : "SILENCE", latency_us);
    }
    
    if (remaining_bytes > 0) {
        printf("[INFO] Ignored %" PRIu32
               " trailing audio bytes (< one inference chunk)\n",
               remaining_bytes);
    }

    free(chunk);
    ret = audio_wav_close(wav);
    if (ret != ESP_OK) {
        printf("[ERROR] Failed to close WAV file: %s\n", esp_err_to_name(ret));
        return 1;
    }
    
    if (chunk_num > 0) {
        int64_t avg_latency_us = total_time_us / chunk_num;
        printf("\n[STATS] Processed %d chunks, avg latency: %lld us (%.2f ms)\n", 
               chunk_num, avg_latency_us, avg_latency_us / 1000.0f);
    }
    
    return 0;
}

// ============================================================================
// Command: play_wav <filename.wav>
// ============================================================================

static struct {
    struct arg_str *file;
    struct arg_end *end;
} play_wav_args;

static int cmd_play_wav(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void**)&play_wav_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, play_wav_args.end, argv[0]);
        return 1;
    }
    
    const char* wav_path = play_wav_args.file->sval[0];
    
    audio_wav_handle_t wav = NULL;
    esp_err_t ret = audio_wav_open(wav_path, &wav);
    if (ret != ESP_OK) {
        printf("[ERROR] Failed to open WAV file: %s (%s)\n",
               wav_path, esp_err_to_name(ret));
        return 1;
    }
    
    audio_wav_info_t wav_info {};
    ret = audio_wav_get_info(wav, &wav_info);
    if (ret != ESP_OK) {
        printf("[ERROR] Failed to read WAV metadata: %s\n", esp_err_to_name(ret));
        audio_wav_close(wav);
        return 1;
    }
    if (wav_info.sample_rate != PULSEVAD_SAMPLE_RATE ||
        wav_info.num_channels != PULSEVAD_NUM_CHANNELS ||
        wav_info.bits_per_sample != PULSEVAD_BITS_PER_SAMPLE ||
        wav_info.data_size == 0) {
        printf("[ERROR] Playback requires non-empty %d Hz mono PCM16 WAV\n",
               PULSEVAD_SAMPLE_RATE);
        audio_wav_close(wav);
        return 1;
    }
    
    printf("[INFO] Playing: %s\n", wav_path);
    printf("       Format: %" PRIu32 " Hz, %u ch, %u bit\n",
           wav_info.sample_rate, (unsigned)wav_info.num_channels,
           (unsigned)wav_info.bits_per_sample);
    
    ret = audio_manager_start_playback();
    if (ret != ESP_OK) {
        printf("[ERROR] Failed to start playback: %s\n", esp_err_to_name(ret));
        audio_wav_close(wav);
        return 1;
    }
    
    int16_t buffer[1024];
    size_t remaining = wav_info.data_size;
    while (remaining > 0) {
        const size_t requested = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
        size_t bytes_read = 0;
        ret = audio_wav_read(wav, buffer, requested, &bytes_read);
        if (ret != ESP_OK || bytes_read != requested) {
            if (ret == ESP_OK) {
                ret = ESP_ERR_INVALID_RESPONSE;
            }
            printf("[ERROR] Failed to read WAV audio data: %s\n", esp_err_to_name(ret));
            break;
        }

        const int samples_read = (int)(bytes_read / sizeof(int16_t));
        const int samples_written = audio_manager_write(buffer, samples_read, 1000);
        if (samples_written != samples_read) {
            printf("[ERROR] Failed to write WAV audio to speaker\n");
            ret = ESP_FAIL;
            break;
        }
        remaining -= bytes_read;
    }
    
    vTaskDelay(pdMS_TO_TICKS(100));
    const esp_err_t stop_ret = audio_manager_stop_playback();
    if (ret == ESP_OK && stop_ret != ESP_OK) {
        ret = stop_ret;
    }
    const esp_err_t close_ret = audio_wav_close(wav);
    if (ret == ESP_OK && close_ret != ESP_OK) {
        ret = close_ret;
    }
    if (ret != ESP_OK) {
        printf("[ERROR] WAV playback failed: %s\n", esp_err_to_name(ret));
        return 1;
    }

    printf("[OK] Playback finished\n");
    
    return 0;
}

// ============================================================================
// Command: record_mic <output.wav> <duration_sec>
// ============================================================================

static struct {
    struct arg_str *file;
    struct arg_int *duration;
    struct arg_end *end;
} record_mic_args;

static int cmd_record_mic(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void**)&record_mic_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, record_mic_args.end, argv[0]);
        return 1;
    }
    
    const char* output_path = record_mic_args.file->sval[0];
    int duration_sec = record_mic_args.duration->ival[0];
    
    if (duration_sec <= 0 || duration_sec > MAX_AUDIO_DURATION_SEC) {
        printf("[ERROR] Duration must be between 1 and %d seconds\n", MAX_AUDIO_DURATION_SEC);
        return 1;
    }

    int16_t *buffer = (int16_t *)malloc(PULSEVAD_CHUNK_BYTES);
    if (!buffer) {
        printf("[ERROR] Failed to allocate recording buffer\n");
        return 1;
    }
    
    FILE* f = fopen(output_path, "wb");
    if (!f) {
        printf("[ERROR] Failed to create file: %s\n", output_path);
        free(buffer);
        return 1;
    }
    
    esp_err_t ret = write_wav_header(f, 0);
    if (ret != ESP_OK) {
        printf("[ERROR] Failed to write WAV header: %s\n", esp_err_to_name(ret));
        fclose(f);
        free(buffer);
        return 1;
    }
    
    uint32_t total_samples = duration_sec * PULSEVAD_SAMPLE_RATE;
    uint32_t samples_recorded = 0;
    
    printf("[INFO] Recording %d seconds to: %s\n", duration_sec, output_path);
    printf("       Press Ctrl+C to stop early\n");
    
    audio_manager_start_capture();
    
    while (samples_recorded < total_samples) {
        int samples_to_read = (total_samples - samples_recorded) < PULSEVAD_CHUNK_SAMPLES 
                            ? (total_samples - samples_recorded) 
                            : PULSEVAD_CHUNK_SAMPLES;
        
        int read = audio_manager_read(buffer, samples_to_read, 1000);
        if (read > 0) {
            fwrite(buffer, sizeof(int16_t), read, f);
            samples_recorded += read;
            
            if (samples_recorded % (PULSEVAD_SAMPLE_RATE / 2) == 0) {
                printf("\r[RECORDING] %" PRIu32 "/%d seconds...",
                       samples_recorded / PULSEVAD_SAMPLE_RATE, duration_sec);
                fflush(stdout);
            }
        }
    }
    
    printf("\n");
    audio_manager_stop_capture();
    
    // Write proper WAV header
    ret = write_wav_header(f, samples_recorded * sizeof(int16_t));
    
    fclose(f);
    free(buffer);
    if (ret != ESP_OK) {
        printf("[ERROR] Failed to finalize WAV header: %s\n", esp_err_to_name(ret));
        return 1;
    }
    printf("[OK] Recording saved: %s (%" PRIu32 " samples, %.1f sec)\n",
           output_path, samples_recorded, (float)samples_recorded / PULSEVAD_SAMPLE_RATE);
    
    return 0;
}

// ============================================================================
// Command: vad_infer_mic <duration_sec>
// ============================================================================

static struct {
    struct arg_int *duration;
    struct arg_end *end;
} vad_infer_mic_args;

static int cmd_vad_infer_mic(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void**)&vad_infer_mic_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, vad_infer_mic_args.end, argv[0]);
        return 1;
    }

    if (ensure_vad_initialized() != ESP_OK) {
        printf("[ERROR] PulseVAD engine initialization failed\n");
        return 1;
    }

    const int duration_sec = vad_infer_mic_args.duration->ival[0];
    if (duration_sec <= 0 || duration_sec > MAX_AUDIO_DURATION_SEC) {
        printf("[ERROR] Duration must be between 1 and %d seconds\n",
               MAX_AUDIO_DURATION_SEC);
        return 1;
    }

    int16_t *chunk = (int16_t *)malloc(PULSEVAD_CHUNK_BYTES);
    if (!chunk) {
        printf("[ERROR] Failed to allocate microphone inference buffer\n");
        return 1;
    }

    pulsevad_reset(g_vad_handle);
    esp_err_t ret = audio_manager_start_capture();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start microphone capture: %s", esp_err_to_name(ret));
        free(chunk);
        return 1;
    }

    printf("\nLive PulseVAD microphone test (%s, %d seconds)\n",
           audio_manager_get_mic() == MIC_I2S1_INMP441 ? "INMP441" : "ES8311",
           duration_sec);
    printf("Time (s) | Probability | Decision\n");
    printf("---------+-------------+----------\n");

    const int total_chunks = duration_sec * PULSEVAD_SAMPLE_RATE / PULSEVAD_CHUNK_SAMPLES;
    int processed_chunks = 0;
    int speech_chunks = 0;

    for (; processed_chunks < total_chunks; ++processed_chunks) {
        int samples_read = audio_manager_read(
            chunk, PULSEVAD_CHUNK_SAMPLES, BOARD_AUDIO_I2S_READ_TIMEOUT_MS);
        if (samples_read != PULSEVAD_CHUNK_SAMPLES) {
            ESP_LOGE(TAG, "Microphone read returned %d samples; expected %d",
                     samples_read, PULSEVAD_CHUNK_SAMPLES);
            break;
        }

        float probability = 0.0f;
        bool is_speech = false;
        ret = pulsevad_process_i16_chunk(g_vad_handle, chunk, &probability, &is_speech);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "PulseVAD inference failed: %s", esp_err_to_name(ret));
            break;
        }

        speech_chunks += is_speech ? 1 : 0;
        printf("%8.1f | %11.6f | %s\n",
               (processed_chunks + 1) * (float)PULSEVAD_CHUNK_MS / 1000.0f,
               probability, is_speech ? "SPEECH" : "NON_SPEECH");
    }

    esp_err_t stop_ret = audio_manager_stop_capture();
    if (stop_ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to stop microphone capture: %s", esp_err_to_name(stop_ret));
    }

    free(chunk);
    printf("\n[STATS] Processed %d/%d chunks; speech decisions: %d (threshold %.2f)\n",
           processed_chunks, total_chunks, speech_chunks, g_threshold);
    return processed_chunks == total_chunks && stop_ret == ESP_OK ? 0 : 1;
}

// ============================================================================
// Command: selftest
// ============================================================================

static int cmd_golden(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const esp_err_t ret = pulsevad_run_self_test();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "PulseVAD golden inference test failed: %s", esp_err_to_name(ret));
        return 1;
    }

    printf("[PASS] PulseVAD golden inference vectors\n");
    return 0;
}

static int cmd_selftest(int argc, char **argv) {
    printf("\n=== PulseVAD Self-Test ===\n");
    
    // Test 1: Audio Manager
    printf("[1/3] Audio Manager: ");
    if (audio_manager_is_initialized()) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        return 1;
    }
    
    // Test 2: SD card mount
    printf("[2/3] SD Card Mount: ");
    struct stat st;
    if (stat(FS_MOUNT_POINT_SD, &st) == 0 && S_ISDIR(st.st_mode)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    
    // Test 3: ESPM model availability
    char model_2k[MAX_FILE_PATH];
    snprintf(model_2k, sizeof(model_2k), "%s/models/pulsevad_2.1k.espm",
             FS_MOUNT_POINT_SD);
    printf("[3/3] Student ESPM model: ");
    if (stat(model_2k, &st) == 0) {
        printf("PASS (pulsevad_2.1k.espm)\n");
    } else {
        printf("not installed (ROM fallback remains available)\n");
    }
    
    printf("==========================\n\n");
    return 0;
}

// ============================================================================
// Command Registration
// ============================================================================

void cmd_vad_cli_register(void) {
    // ls
    ls_args.path = arg_str0(NULL, NULL, "<path>", "Directory path (default: /sd/models)");
    ls_args.end = arg_end(1);
    register_command("info", "Display system information", &cmd_info);
    register_command("model", "Display the active VAD model", &cmd_model);
    register_command("ls", "List directory contents", &cmd_ls, &ls_args);

    // load_model
    load_model_args.path = arg_str1(NULL, NULL, "<path.espm>", "ESPM model file path");
    load_model_args.end = arg_end(1);
    register_command("load_model", "Load PulseVAD model from SD", &cmd_load_model, &load_model_args);
    register_command("unload_model", "Unload current model", &cmd_unload_model);

    // threshold
    threshold_args.value = arg_dbl1(NULL, NULL, "<0.0-1.0>", "Speech threshold");
    threshold_args.end = arg_end(1);
    register_command("threshold", "Set speech detection threshold (0.0-1.0)", &cmd_threshold, &threshold_args);

    register_command("mic", "Show, select, or configure microphone input", &cmd_mic);

    // speaker_vol
    speaker_vol_args.volume = arg_int1(NULL, NULL, "<0-100>", "Volume level");
    speaker_vol_args.end = arg_end(1);
    register_command("speaker_vol", "Set speaker volume (0-100%)", &cmd_speaker_vol, &speaker_vol_args);

    // test_wav
    test_wav_args.file = arg_str1(NULL, NULL, "<file.wav>", "WAV file path");
    test_wav_args.end = arg_end(1);
    register_command("test_wav", "Test VAD on WAV file (16kHz mono 16-bit)", &cmd_test_wav, &test_wav_args);

    // play_wav
    play_wav_args.file = arg_str1(NULL, NULL, "<file.wav>", "WAV file path");
    play_wav_args.end = arg_end(1);
    register_command("play_wav", "Play WAV file through speaker", &cmd_play_wav, &play_wav_args);

    // record_mic
    record_mic_args.file = arg_str1(NULL, NULL, "<output.wav>", "Output file path");
    record_mic_args.duration = arg_int1(NULL, NULL, "<seconds>", "Recording duration");
    record_mic_args.end = arg_end(2);
    register_command("record_mic", "Record from microphone to WAV file", &cmd_record_mic, &record_mic_args);

    // vad_infer_mic
    vad_infer_mic_args.duration = arg_int1(NULL, NULL, "<seconds>", "Inference duration");
    vad_infer_mic_args.end = arg_end(1);
    register_command("vad_infer_mic", "Run live PulseVAD inference on the selected microphone",
                     &cmd_vad_infer_mic, &vad_infer_mic_args);

    register_command("selftest", "Run system self-test", &cmd_selftest);
    register_command("golden", "Run PulseVAD golden inference vectors", &cmd_golden);

    ESP_LOGI(TAG, "All console commands registered");
}
