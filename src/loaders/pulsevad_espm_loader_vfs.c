#include "pulsevad_espm_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "esp_model_format.h"
#include "esp_log.h"

static const char *TAG = "PULSEVAD_ESPM_VFS";

esp_err_t pulsevad_load_model_espm_file(pulsevad_handle_t handle,
                                        const char *espm_path)
{
    if (!handle || !espm_path) {
        return ESP_ERR_INVALID_ARG;
    }

    struct stat file_stat;
    if (stat(espm_path, &file_stat) != 0) {
        ESP_LOGE(TAG, "Cannot stat ESPM model: %s", espm_path);
        return ESP_ERR_NOT_FOUND;
    }
    if (file_stat.st_size < ESPM_HEADER_SIZE ||
        (uint64_t)file_stat.st_size > PULSEVAD_ESPM_MAX_FILE_SIZE) {
        ESP_LOGE(TAG, "ESPM model size is invalid: %ld bytes",
                 (long)file_stat.st_size);
        return ESP_ERR_INVALID_SIZE;
    }

    const size_t file_size = (size_t)file_stat.st_size;
    FILE *file = fopen(espm_path, "rb");
    if (!file) {
        ESP_LOGE(TAG, "Cannot open ESPM model: %s", espm_path);
        return ESP_ERR_NOT_FOUND;
    }

    uint8_t *raw_data = (uint8_t *)malloc(file_size);
    if (!raw_data) {
        fclose(file);
        return ESP_ERR_NO_MEM;
    }

    const size_t read_size = fread(raw_data, 1, file_size, file);
    const int close_result = fclose(file);
    if (read_size != file_size || close_result != 0) {
        ESP_LOGE(TAG, "Failed to read ESPM model completely");
        free(raw_data);
        return ESP_FAIL;
    }

    return pulsevad_load_model_espm_buffer(handle, raw_data, file_size);
}
