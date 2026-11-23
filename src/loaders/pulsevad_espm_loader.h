#pragma once

#include "esp_pulsevad.h"

#include <stddef.h>
#include <stdint.h>

#define PULSEVAD_ESPM_MAX_FILE_SIZE (1024U * 1024U)

/* Takes ownership of raw_data on every return path. */
esp_err_t pulsevad_load_model_espm_buffer(pulsevad_handle_t handle,
                                          uint8_t *raw_data,
                                          size_t file_size);
