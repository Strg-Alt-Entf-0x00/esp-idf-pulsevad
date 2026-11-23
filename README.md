# PulseVAD for ESP-IDF

PulseVAD is a compact, single-precision voice activity detector (VAD) component
for ESP32-P4 applications. It classifies mono audio in fixed 200 ms windows
and provides a stateful speech decision with configurable threshold and
hysteresis.

The built-in Student model runs without an external model file. Applications
can optionally load a Student or Teacher model from their mounted filesystem.

## Supported configuration

- ESP32-P4
- ESP-IDF 6.0.2 or later
- Mono, 16 kHz audio
- Exactly 3,200 samples per processing call (200 ms)
- Float32 samples normalized to `[-1.0, 1.0]`, or signed 16-bit PCM

PulseVAD does not capture audio, resample input, or mount filesystems. Those
tasks remain the application's responsibility.

## Add PulseVAD to an ESP-IDF project

Place this component checkout in your application's `components` directory.
Set the component search path before including ESP-IDF's project CMake file:

```cmake
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/components/esp-idf-pulsevad")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
```

Declare the component as a dependency in the application component:

```cmake
idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES esp-idf-pulsevad
)
```

## Basic use

Include `esp_pulsevad.h`, initialize one handle, and pass exactly one
3,200-sample window to each processing call:

```c
#include "esp_pulsevad.h"

pulsevad_config_t config = pulsevad_default_config();
pulsevad_handle_t vad = NULL;
ESP_ERROR_CHECK(pulsevad_init(&config, &vad));

float probability = 0.0f;
bool is_speech = false;
ESP_ERROR_CHECK(pulsevad_process_i16_chunk(
    vad, pcm_3200, &probability, &is_speech));

pulsevad_free(vad);
```

`pcm_3200` must point to 3,200 contiguous `int16_t` samples at 16 kHz.
`pulsevad_process_chunk()` accepts the equivalent float32 input. The output
probability is the model score; the speech decision also applies the handle's
temporal hysteresis state.

## Optional model files

Optional model files use the ESPM container format and are validated through
the [`esp-idf-model-manager`](https://github.com/Strg-Alt-Entf-0x00/esp-idf-model-manager).
PulseVAD owns model loading and inference; the model manager provides the ESPM
format parsing and validation used by the component.

Mount the application's filesystem first, then load a model with
`pulsevad_load_model_espm_file()`. The component does not choose or mount a
filesystem. Resetting the handle to the built-in model is available through
`pulsevad_reset_to_default_model()`.

The built-in Student model is part of the firmware and needs no downloaded
file. Optional model download and conversion utilities are provided under
[`tools-py/`](tools-py/); they are not required to build or use the built-in
model.

## Board example

The [ESP32-P4 board example](examples/ws-esp32-p4-wifi6-console-vad-dev/README.md)
documents audio input, SD-card storage, networking, console commands, and
board-specific build and flash steps.

## Hardware validation

The component and board application were validated on a Waveshare
ESP32-P4-WIFI6 Rev. 1.3 at 360 MHz with ESP-IDF 6.0.2. On the video-disjoint
144,000-window AVA development split, all ROM-Student, ESPM-Student, and
ESPM-Teacher decisions matched their original FP32 references with zero
mismatches.

| Model | Mean latency per 200 ms chunk |
|---|---:|
| ROM Student | 3.270 ms |
| ESPM Student | 3.273 ms |
| ESPM Teacher | 15.889 ms |

These measurements describe a specific board, firmware, corpus, and run. They
are not general performance or latency guarantees.

## Licensing

The root [`LICENSE`](LICENSE) contains the PulseVAD source license and a
separate attribution and license notice for the upstream model artifacts.

---

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
- **Free for noncommercial use**: Personal, educational, research, and hobbyist projects are completely free.
- **Commercial use**: Not permitted without prior authorization. For commercial inquiries, please contact the author via GitHub.
