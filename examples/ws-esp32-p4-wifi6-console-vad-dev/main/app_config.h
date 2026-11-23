#ifndef PULSEVAD_CONSOLE_DEV_CONFIG_H
#define PULSEVAD_CONSOLE_DEV_CONFIG_H

#include "esp_pulsevad.h"

#define PULSEVAD_SAMPLE_RATE          PULSEVAD_SAMPLE_RATE_HZ
#define PULSEVAD_BITS_PER_SAMPLE      16
#define PULSEVAD_NUM_CHANNELS         1
#define PULSEVAD_BYTES_PER_SAMPLE     2
#define PULSEVAD_CHUNK_MS              200
#define PULSEVAD_CHUNK_BYTES           (PULSEVAD_CHUNK_SAMPLES * PULSEVAD_BYTES_PER_SAMPLE)

#define DEFAULT_VAD_THRESHOLD          0.50f
#define MIN_VAD_THRESHOLD              0.0f
#define MAX_VAD_THRESHOLD              1.0f

#define FS_MOUNT_POINT_SD              "/sd"
#define MODEL_FILE_EXTENSION           ".espm"
#define WAV_FILE_EXTENSION             ".wav"

#define MAX_FILE_PATH                  256
#define MAX_FILENAME                   64
#define MAX_AUDIO_DURATION_SEC         300
#define MAX_AUDIO_SAMPLES              (MAX_AUDIO_DURATION_SEC * PULSEVAD_SAMPLE_RATE)
#define MAX_AUDIO_BYTES                (MAX_AUDIO_SAMPLES * PULSEVAD_BYTES_PER_SAMPLE)
#define MAX_WAV_FILE_SIZE              (100 * 1024 * 1024)
#define MAX_MODEL_FILE_SIZE            (1 * 1024 * 1024)

#define CONSOLE_MAX_COMMAND_LINE       256
#define CONSOLE_PROMPT                 "pulsevad_dev> "
#define CONSOLE_HISTORY_SIZE           30

#endif
