#include "esp_pulsevad.h"
#include "esp_model_format.h"
#include "pulsevad_internal.h"
#include "pulsevad_espm_loader.h"
#include "esp_log.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PULSEVAD_ESPM_LAYER_COUNT 14U

static const char *TAG = "PULSEVAD_ESPM";

typedef struct {
    const uint8_t *data;
    size_t size;
} pulsevad_format_buffer_t;

typedef struct {
    const char *name;
    uint16_t shape[3];
    uint16_t bias_count;
} pulsevad_layer_spec_t;

static const pulsevad_layer_spec_t s_layer_specs[2][PULSEVAD_ESPM_LAYER_COUNT] = {
    {
        {"adapter",    {12, 64, 1}, 12},
        {"conv0_dw",   {12, 1, 11}, 0},
        {"conv0_pw",   {8, 12, 1}, 8},
        {"block1",     {8, 8, 1}, 8},
        {"block2",     {8, 8, 1}, 8},
        {"subA_dw",    {8, 1, 17}, 0},
        {"subA_pw",    {8, 8, 1}, 8},
        {"subC_dw",    {8, 1, 17}, 0},
        {"subC_pw",    {8, 8, 1}, 8},
        {"skip",       {8, 8, 1}, 8},
        {"conv4_dw",   {8, 1, 29}, 0},
        {"conv4_pw",   {8, 8, 1}, 8},
        {"conv5",      {8, 8, 1}, 8},
        {"classifier", {2, 8, 0}, 2},
    },
    {
        {"adapter",    {128, 64, 1}, 128},
        {"conv0_dw",   {128, 1, 11}, 0},
        {"conv0_pw",   {128, 128, 1}, 128},
        {"block1",     {64, 128, 1}, 64},
        {"block2",     {64, 64, 1}, 64},
        {"subA_dw",    {64, 1, 17}, 0},
        {"subA_pw",    {64, 64, 1}, 64},
        {"subC_dw",    {64, 1, 17}, 0},
        {"subC_pw",    {64, 64, 1}, 64},
        {"skip",       {64, 64, 1}, 64},
        {"conv4_dw",   {64, 1, 29}, 0},
        {"conv4_pw",   {128, 64, 1}, 128},
        {"conv5",      {128, 128, 1}, 128},
        {"classifier", {2, 128, 0}, 2},
    },
};

static esp_err_t pulsevad_buffer_read_at(void *context, uint64_t offset,
                                         void *buffer, size_t size)
{
    const pulsevad_format_buffer_t *source = context;
    if (!source || (!buffer && size != 0) || offset > source->size ||
        size > source->size - (size_t)offset) {
        return ESP_ERR_INVALID_SIZE;
    }
    memcpy(buffer, source->data + (size_t)offset, size);
    return ESP_OK;
}

static const esp_model_section_t *find_container_section(
    const esp_model_container_t *container, uint16_t type)
{
    for (uint16_t i = 0; i < container->header.num_sections; ++i) {
        if (container->sections[i].section_type == type) {
            return &container->sections[i];
        }
    }
    return NULL;
}

static bool floats_finite(const uint8_t *data, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        float value;
        memcpy(&value, data + i * sizeof(float), sizeof(value));
        if (!isfinite(value)) {
            return false;
        }
    }
    return true;
}

static void assign_layer_pointer(pulsevad_dynamic_model_t *model, pulsevad_layer_t *layer)
{
    if (strcmp(layer->name, "adapter") == 0) model->adapter = layer;
    else if (strcmp(layer->name, "conv0_dw") == 0) model->conv0_dw = layer;
    else if (strcmp(layer->name, "conv0_pw") == 0) model->conv0_pw = layer;
    else if (strcmp(layer->name, "block1") == 0) model->block1 = layer;
    else if (strcmp(layer->name, "block2") == 0) model->block2 = layer;
    else if (strcmp(layer->name, "subA_dw") == 0) model->subA_dw = layer;
    else if (strcmp(layer->name, "subA_pw") == 0) model->subA_pw = layer;
    else if (strcmp(layer->name, "subC_dw") == 0) model->subC_dw = layer;
    else if (strcmp(layer->name, "subC_pw") == 0) model->subC_pw = layer;
    else if (strcmp(layer->name, "skip") == 0) model->skip = layer;
    else if (strcmp(layer->name, "conv4_dw") == 0) model->conv4_dw = layer;
    else if (strcmp(layer->name, "conv4_pw") == 0) model->conv4_pw = layer;
    else if (strcmp(layer->name, "conv5") == 0) model->conv5 = layer;
    else if (strcmp(layer->name, "classifier") == 0) model->classifier = layer;
}

static esp_err_t metadata_required_value(
    const esp_model_metadata_view_t *metadata, const char *key,
    esp_model_metadata_type_t expected_type,
    esp_model_metadata_value_t *out_value)
{
    const esp_err_t error = esp_model_metadata_find(metadata, key, out_value);
    if (error != ESP_OK) {
        return error;
    }
    if (out_value->type != expected_type ||
        (out_value->flags & ESPM_METADATA_FLAG_REQUIRED) == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    return ESP_OK;
}

static esp_err_t metadata_required_u32(const esp_model_metadata_view_t *metadata,
                                       const char *key, uint32_t *out_value)
{
    esp_model_metadata_value_t value;
    const esp_err_t error = metadata_required_value(
        metadata, key, ESPM_METADATA_UINT32, &value);
    if (error != ESP_OK) {
        return error;
    }
    memcpy(out_value, value.value, sizeof(*out_value));
    return ESP_OK;
}

static esp_err_t metadata_required_f32(const esp_model_metadata_view_t *metadata,
                                       const char *key, float *out_value)
{
    esp_model_metadata_value_t value;
    const esp_err_t error = metadata_required_value(
        metadata, key, ESPM_METADATA_FLOAT32, &value);
    if (error != ESP_OK) {
        return error;
    }
    memcpy(out_value, value.value, sizeof(*out_value));
    return isfinite(*out_value) ? ESP_OK : ESP_ERR_INVALID_ARG;
}

static esp_err_t metadata_required_string(const esp_model_metadata_view_t *metadata,
                                          const char *key, const char **out_value,
                                          size_t *out_size)
{
    esp_model_metadata_value_t value;
    const esp_err_t error = metadata_required_value(
        metadata, key, ESPM_METADATA_UTF8, &value);
    if (error != ESP_OK) {
        return error;
    }
    *out_value = (const char *)value.value;
    *out_size = value.value_size;
    return ESP_OK;
}

static bool metadata_string_equals(const esp_model_metadata_view_t *metadata,
                                   const char *key, const char *expected)
{
    const char *value;
    size_t value_size;
    const size_t expected_size = strlen(expected);
    return metadata_required_string(metadata, key, &value, &value_size) == ESP_OK &&
           value_size == expected_size &&
           memcmp(value, expected, expected_size) == 0;
}

static bool metadata_sha256_valid(const esp_model_metadata_view_t *metadata,
                                  const char *key)
{
    const char *value;
    size_t value_size;
    if (metadata_required_string(metadata, key, &value, &value_size) != ESP_OK ||
        value_size != 64) {
        return false;
    }
    for (size_t i = 0; i < value_size; ++i) {
        const char ch = value[i];
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
              (ch >= 'A' && ch <= 'F'))) {
            return false;
        }
    }
    return true;
}

static const esp_model_tensor_record_v2_t *find_tensor_record(
    const esp_model_tensor_record_v2_t *records, uint32_t record_count,
    const char *name, uint32_t *out_index)
{
    for (uint32_t i = 0; i < record_count; ++i) {
        if (records[i].scope_id == 0 && strcmp(records[i].name, name) == 0) {
            if (out_index) {
                *out_index = i;
            }
            return &records[i];
        }
    }
    return NULL;
}

static bool tensor_matches(const esp_model_tensor_record_v2_t *record,
                           const uint32_t *shape, uint8_t rank,
                           const uint8_t *tensor_data, size_t tensor_data_size)
{
    if (!record || record->dtype != ESPM_TENSOR_DTYPE_FP32 ||
        record->quant_type != ESPM_TENSOR_QUANT_NONE ||
        record->rank != rank || record->data_size > SIZE_MAX ||
        record->data_offset > tensor_data_size ||
        record->data_size > tensor_data_size - (size_t)record->data_offset) {
        return false;
    }
    uint64_t element_count = 1;
    for (uint8_t axis = 0; axis < rank; ++axis) {
        if (record->shape[axis] != shape[axis]) {
            return false;
        }
        element_count *= shape[axis];
    }
    return element_count == record->num_elements &&
           record->data_size == element_count * sizeof(float) &&
           floats_finite(tensor_data + (size_t)record->data_offset,
                         (size_t)element_count);
}

static esp_err_t parse_model_v3(const uint8_t *raw_data,
                                size_t raw_size,
                                const esp_model_header_t *header,
                                const esp_model_section_t *metadata_section,
                                const esp_model_section_t *index_section,
                                const esp_model_section_t *tensor_data_section,
                                pulsevad_dynamic_model_t *model)
{
    if (!raw_data || !header || !metadata_section || !index_section ||
        !tensor_data_section || !model ||
        metadata_section->size > ESPM_MAX_METADATA_SIZE ||
        metadata_section->size > SIZE_MAX ||
        index_section->size > SIZE_MAX ||
        tensor_data_section->size > SIZE_MAX) {
        return ESP_ERR_INVALID_SIZE;
    }
    const uint8_t *metadata_data = raw_data + (size_t)metadata_section->offset;
    const uint8_t *index_data = raw_data + (size_t)index_section->offset;
    const uint8_t *tensor_data = raw_data + (size_t)tensor_data_section->offset;
    esp_model_metadata_view_t metadata;
    esp_err_t error = esp_model_metadata_open(
        metadata_data, (size_t)metadata_section->size, &metadata);
    if (error != ESP_OK) {
        return error;
    }

    uint32_t metadata_key_capacity = 32U + PULSEVAD_NUM_MELS * 2U;
    const char **known_keys = calloc(metadata_key_capacity, sizeof(*known_keys));
    char (*mel_keys)[48] = calloc(PULSEVAD_NUM_MELS * 2U, sizeof(*mel_keys));
    if (!known_keys || !mel_keys) {
        free(known_keys);
        free(mel_keys);
        return ESP_ERR_NO_MEM;
    }
    size_t known_key_count = 0;
    static const char *const fixed_keys[] = {
        "general.architecture",
        "general.architecture_version",
        "pulsevad.variant",
        "pulsevad.total_params",
        "pulsevad.prior_bias_shift",
        "pulsevad.checkpoint_sha256",
        "pulsevad.mel_filterbank_sha256",
        "pulsevad.mel_filter_count",
        "pulsevad.frontend.sample_rate",
        "pulsevad.frontend.window_samples",
        "pulsevad.frontend.n_fft",
        "pulsevad.frontend.win_length",
        "pulsevad.frontend.hop_length",
        "pulsevad.frontend.num_frames",
        "pulsevad.frontend.num_mels",
        "pulsevad.frontend.preemphasis_alpha",
        "pulsevad.frontend.epsilon",
    };
    for (size_t i = 0; i < sizeof(fixed_keys) / sizeof(fixed_keys[0]); ++i) {
        known_keys[known_key_count++] = fixed_keys[i];
    }
    for (size_t i = 0; i < PULSEVAD_NUM_MELS; ++i) {
        snprintf(mel_keys[i * 2], sizeof(mel_keys[i * 2]),
                 "pulsevad.mel.%02u.start_k", (unsigned)i);
        snprintf(mel_keys[i * 2 + 1], sizeof(mel_keys[i * 2 + 1]),
                 "pulsevad.mel.%02u.count", (unsigned)i);
        known_keys[known_key_count++] = mel_keys[i * 2];
        known_keys[known_key_count++] = mel_keys[i * 2 + 1];
    }
    error = esp_model_metadata_validate_required_keys(
        &metadata, known_keys, known_key_count);
    free(known_keys);
    free(mel_keys);
    if (error != ESP_OK) {
        return error;
    }

    uint32_t architecture_version;
    uint32_t variant_index;
    uint32_t total_params;
    uint32_t mel_filter_count;
    float prior_bias_shift;
    if (!metadata_string_equals(&metadata, "general.architecture", "pulsevad.cnn") ||
        metadata_required_u32(&metadata, "general.architecture_version",
                              &architecture_version) != ESP_OK ||
        architecture_version != 1 ||
        metadata_required_u32(&metadata, "pulsevad.total_params",
                              &total_params) != ESP_OK ||
        metadata_required_u32(&metadata, "pulsevad.mel_filter_count",
                              &mel_filter_count) != ESP_OK ||
        mel_filter_count != PULSEVAD_NUM_MELS ||
        metadata_required_f32(&metadata, "pulsevad.prior_bias_shift",
                             &prior_bias_shift) != ESP_OK ||
        !metadata_sha256_valid(&metadata, "pulsevad.checkpoint_sha256") ||
        !metadata_sha256_valid(&metadata, "pulsevad.mel_filterbank_sha256")) {
        return ESP_ERR_NOT_SUPPORTED;
    }

    const char *variant;
    size_t variant_size;
    if (metadata_required_string(&metadata, "pulsevad.variant",
                                 &variant, &variant_size) != ESP_OK) {
        return ESP_ERR_INVALID_ARG;
    }
    if (variant_size == sizeof("student") - 1 &&
        memcmp(variant, "student", sizeof("student") - 1) == 0) {
        variant_index = 0;
    } else if (variant_size == sizeof("teacher") - 1 &&
               memcmp(variant, "teacher", sizeof("teacher") - 1) == 0) {
        variant_index = 1;
    } else {
        return ESP_ERR_NOT_SUPPORTED;
    }

    static const uint32_t frontend_integer_values[] = {
        PULSEVAD_SAMPLE_RATE, PULSEVAD_WINDOW_SAMPLES, PULSEVAD_N_FFT,
        PULSEVAD_WIN_LENGTH, PULSEVAD_HOP_LENGTH, PULSEVAD_NUM_FRAMES,
        PULSEVAD_NUM_MELS,
    };
    static const char *const frontend_integer_keys[] = {
        "pulsevad.frontend.sample_rate",
        "pulsevad.frontend.window_samples",
        "pulsevad.frontend.n_fft",
        "pulsevad.frontend.win_length",
        "pulsevad.frontend.hop_length",
        "pulsevad.frontend.num_frames",
        "pulsevad.frontend.num_mels",
    };
    for (size_t i = 0; i < sizeof(frontend_integer_values) /
                            sizeof(frontend_integer_values[0]); ++i) {
        uint32_t value;
        if (metadata_required_u32(&metadata, frontend_integer_keys[i],
                                  &value) != ESP_OK ||
            value != frontend_integer_values[i]) {
            return ESP_ERR_NOT_SUPPORTED;
        }
    }
    float frontend_float;
    if (metadata_required_f32(&metadata,
                              "pulsevad.frontend.preemphasis_alpha",
                              &frontend_float) != ESP_OK ||
        fabsf(frontend_float - PULSEVAD_PREEMPHASIS_ALPHA) >= 1e-7f ||
        metadata_required_f32(&metadata, "pulsevad.frontend.epsilon",
                              &frontend_float) != ESP_OK ||
        fabsf(frontend_float - PULSEVAD_EPS) >= 1e-9f) {
        return ESP_ERR_NOT_SUPPORTED;
    }

    esp_model_tensor_index_header_v2_t index_header;
    const esp_model_tensor_record_v2_t *records = NULL;
    error = esp_model_tensor_index_v2_validate(
        index_data, (size_t)index_section->size,
        tensor_data_section->size, &index_header, &records);
    if (error != ESP_OK) {
        return error;
    }
    if (index_header.scope_count != 1) {
        return ESP_ERR_NOT_SUPPORTED;
    }
    bool used[ESPM_MAX_TENSORS] = {false};
    uint64_t observed_params = 0;
    for (size_t i = 0; i < PULSEVAD_ESPM_LAYER_COUNT; ++i) {
        const pulsevad_layer_spec_t *spec = &s_layer_specs[variant_index][i];
        char tensor_name[64];
        snprintf(tensor_name, sizeof(tensor_name), "layer.%s.weight", spec->name);
        const uint8_t rank = i == PULSEVAD_ESPM_LAYER_COUNT - 1 ? 2 : 3;
        uint32_t shape[3] = {spec->shape[0], spec->shape[1], spec->shape[2]};
        uint32_t record_index;
        const esp_model_tensor_record_v2_t *weight = find_tensor_record(
            records, index_header.tensor_count, tensor_name, &record_index);
        if (!tensor_matches(weight, shape, rank, tensor_data,
                            (size_t)tensor_data_section->size)) {
            return ESP_ERR_INVALID_ARG;
        }
        used[record_index] = true;

        pulsevad_layer_t *layer = &model->layers[i];
        snprintf(layer->name, sizeof(layer->name), "%s", spec->name);
        memcpy(layer->w_shape, spec->shape, sizeof(layer->w_shape));
        layer->has_bias = spec->bias_count != 0;
        layer->num_weights = weight->num_elements;
        layer->weights = (float *)(tensor_data + (size_t)weight->data_offset);
        observed_params += weight->num_elements;

        if (spec->bias_count != 0) {
            snprintf(tensor_name, sizeof(tensor_name), "layer.%s.bias", spec->name);
            const uint32_t bias_shape[1] = {spec->bias_count};
            const esp_model_tensor_record_v2_t *bias = find_tensor_record(
                records, index_header.tensor_count, tensor_name, &record_index);
            if (!tensor_matches(bias, bias_shape, 1, tensor_data,
                                (size_t)tensor_data_section->size)) {
                return ESP_ERR_INVALID_ARG;
            }
            used[record_index] = true;
            layer->num_biases = spec->bias_count;
            layer->biases = (float *)(tensor_data + (size_t)bias->data_offset);
            observed_params += spec->bias_count;
        }
        assign_layer_pointer(model, layer);
    }

    for (size_t i = 0; i < PULSEVAD_NUM_MELS; ++i) {
        char key[48];
        uint32_t start;
        uint32_t count;
        snprintf(key, sizeof(key), "pulsevad.mel.%02u.start_k", (unsigned)i);
        if (metadata_required_u32(&metadata, key, &start) != ESP_OK) {
            return ESP_ERR_INVALID_ARG;
        }
        snprintf(key, sizeof(key), "pulsevad.mel.%02u.count", (unsigned)i);
        if (metadata_required_u32(&metadata, key, &count) != ESP_OK ||
            start > 256 || count > 257 ||
            start + count > PULSEVAD_N_FFT / 2 + 1) {
            return ESP_ERR_INVALID_ARG;
        }
        pulsevad_sparse_mel_bin_t *bin = &model->mel_bins[i];
        bin->start_k = (uint16_t)start;
        bin->num_k = (uint16_t)count;
        if (count == 0) {
            bin->weights = NULL;
            continue;
        }
        char tensor_name[64];
        snprintf(tensor_name, sizeof(tensor_name),
                 "mel.filter.%02u.weights", (unsigned)i);
        const uint32_t mel_shape[1] = {count};
        uint32_t record_index;
        const esp_model_tensor_record_v2_t *weights = find_tensor_record(
            records, index_header.tensor_count, tensor_name, &record_index);
        if (!tensor_matches(weights, mel_shape, 1, tensor_data,
                            (size_t)tensor_data_section->size)) {
            return ESP_ERR_INVALID_ARG;
        }
        used[record_index] = true;
        bin->weights = (const float *)(tensor_data + (size_t)weights->data_offset);
    }
    for (uint32_t i = 0; i < index_header.tensor_count; ++i) {
        if (!used[i]) {
            return ESP_ERR_NOT_SUPPORTED;
        }
    }
    if (observed_params != total_params) {
        return ESP_ERR_INVALID_ARG;
    }
    model->total_params = total_params;
    model->prior_bias_shift = prior_bias_shift;
    (void)raw_size;
    return ESP_OK;
}

esp_err_t pulsevad_load_model_espm_buffer(pulsevad_handle_t handle,
                                          uint8_t *raw_data,
                                          size_t file_size)
{
    if (!handle || !raw_data) {
        free(raw_data);
        return ESP_ERR_INVALID_ARG;
    }
    if (file_size < ESPM_HEADER_SIZE ||
        file_size > PULSEVAD_ESPM_MAX_FILE_SIZE) {
        ESP_LOGE(TAG, "ESPM model size is invalid: %zu bytes", file_size);
        free(raw_data);
        return ESP_ERR_INVALID_SIZE;
    }

    esp_err_t result = ESP_ERR_INVALID_SIZE;
    pulsevad_dynamic_model_t *model = NULL;
    pulsevad_format_buffer_t source = {
        .data = raw_data,
        .size = file_size,
    };
    const esp_model_io_t format_io = {
        .context = (void *)&source,
        .read_at = pulsevad_buffer_read_at,
        .file_size = file_size,
    };
    esp_model_container_t container;
    result = esp_model_container_open(&format_io, &container);
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "Invalid ESPM container: %s", esp_err_to_name(result));
        goto cleanup;
    }
    const esp_model_header_t header = container.header;
    if (header.format_version != ESPM_CURRENT_VERSION ||
        header.model_type != MODEL_TYPE_VAD ||
        header.quant_type != MODEL_QUANT_FP32 ||
        header.sample_rate != PULSEVAD_SAMPLE_RATE) {
        result = ESP_ERR_NOT_SUPPORTED;
        goto cleanup;
    }
    for (uint16_t i = 0; i < header.num_sections; ++i) {
        const esp_model_section_t *section = &container.sections[i];
        if ((section->section_type == SECTION_TYPE_VOCAB ||
             section->section_type == SECTION_TYPE_CONFIG_JSON) &&
            (section->flags & ESPM_SECTION_FLAG_OPTIONAL) == 0) {
            result = ESP_ERR_NOT_SUPPORTED;
            goto cleanup;
        }
    }

    for (uint16_t i = 0; i < header.num_sections; ++i) {
        result = esp_model_container_verify_section_buffer(
            &container.sections[i],
            raw_data + (size_t)container.sections[i].offset,
            (size_t)container.sections[i].size);
        if (result != ESP_OK) {
            ESP_LOGE(TAG, "ESPM section CRC mismatch for %s",
                     container.sections[i].name);
            goto cleanup;
        }
    }

    model = (pulsevad_dynamic_model_t *)calloc(1, sizeof(*model));
    if (!model) {
        result = ESP_ERR_NO_MEM;
        goto cleanup;
    }
    const esp_model_section_t *metadata_section =
        find_container_section(&container, SECTION_TYPE_METADATA);
    const esp_model_section_t *index_section =
        find_container_section(&container, SECTION_TYPE_TENSOR_INDEX);
    const esp_model_section_t *tensor_data_section =
        find_container_section(&container, SECTION_TYPE_TENSOR_DATA);
    if (!metadata_section || !index_section || !tensor_data_section ||
        ((metadata_section->flags | index_section->flags |
          tensor_data_section->flags) & ESPM_SECTION_FLAG_OPTIONAL) != 0) {
        result = ESP_ERR_INVALID_ARG;
        goto cleanup;
    }
    result = parse_model_v3(raw_data, file_size, &header,
                            metadata_section, index_section,
                            tensor_data_section, model);
    if (result != ESP_OK) {
        goto cleanup;
    }

    memcpy(model->model_name, header.model_name, sizeof(model->model_name));
    model->raw_data = raw_data;
    model->raw_size = file_size;
    raw_data = NULL;

    pulsevad_dynamic_model_t *previous_model = handle->dynamic_model;
    handle->dynamic_model = model;
    model = NULL;
    if (previous_model) {
        free(previous_model->raw_data);
        free(previous_model);
    }
    ESP_LOGI(TAG, "Loaded ESPM model '%s' (%lu parameters, %.1f KiB)",
             handle->dynamic_model->model_name,
             (unsigned long)handle->dynamic_model->total_params,
             (float)file_size / 1024.0f);
    return ESP_OK;

cleanup:
    if (model) {
        free(model);
    }
    free(raw_data);
    return result;
}
