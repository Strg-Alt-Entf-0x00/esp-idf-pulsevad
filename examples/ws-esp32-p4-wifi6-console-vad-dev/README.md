# ESP32-P4 Wi-Fi 6 VAD Console Example

This example demonstrates PulseVAD on the Waveshare ESP32-P4-WIFI6 board. It
includes microphone capture, WAV playback and inference, SD-card storage, and
authenticated WLAN file transfer. Board-specific setup and commands are
documented here; the [root README](../../README.md) describes the reusable
PulseVAD component.

The built-in 2,118-parameter Student model is always available. Optional
Student and Teacher model files can be loaded from the SD card. The runtime
accepts ESPM model files only.

## Requirements

- Waveshare ESP32-P4-WIFI6 board
- ESP-IDF 6.0.2 or later
- SD card for optional models and WAV files
- A serial connection for the interactive console

## Build and flash

Install ESP-IDF and make the project's declared component dependencies
available to the ESP-IDF Component Manager. From this example's project
directory, run:

```text
idf.py build
idf.py -p PORT flash
idf.py -p PORT monitor
```

Replace `PORT` with the serial port for the board. The `esp-build.bat` script
is an optional Windows convenience wrapper. The application partition table
contains NVS and firmware; optional model and WAV files are stored on the SD
card.

## Storage and WLAN transfer

The application mounts the SD card at `/sd` and creates these directories:

- Models: `/sd/models/`
- WAV inputs and recordings: `/sd/audio/`
- Test results: `/sd/results/`

WLAN uses ESP-Hosted over SDIO with the board's ESP32-C6 co-processor.
Credentials are stored in NVS and are not printed to the console. At the
console, run:

```text
wifi setup
wifi status
wifi token
```

`wifi setup` prompts for the network name and password, stores them in NVS,
and starts the station. After `wifi status` reports an IP address, use the
one-time-per-boot transfer token shown by `wifi token`. The HTTP service
requires the token and restricts file operations to the models, audio, and
results directories.

Upload and download model files through the `/api/models/` endpoint:

```text
curl --request PUT --header "Authorization: Bearer TOKEN" --data-binary '@pulsevad_2.1k.espm' http://DEVICE_IP:8080/api/models/pulsevad_2.1k.espm
curl --header "Authorization: Bearer TOKEN" http://DEVICE_IP:8080/api/models/pulsevad_2.1k.espm --output pulsevad_2.1k.espm
```

Replace `TOKEN` and `DEVICE_IP` with the values for the running board. On
Windows PowerShell, invoke `curl.exe`. Uploads do not overwrite existing
files; remove or rename a file before uploading its replacement.

## Optional models

The built-in Student model requires no downloaded files. Optional model
files can be prepared from the repository root with Python:

```text
python tools-py/pulsevad_models_py_downloader/pulsevad_download_models_py.py
python tools-py/pulsevad_models_converter/pulsevad_convert_models.py
```

The downloader saves the pinned source checkpoints and Mel filterbank under
`models/py/`, where the converter reads them. The converter writes the ESPM
files into the repository's `models/espm/` directory:

- `pulsevad_2.1k.espm` — Student
- `pulsevad_teacher_81k.espm` — Teacher

The downloader requires Python with `pip`; the converter requires PyTorch and
NumPy. Neither step is needed to use the built-in Student model.

After uploading a file, load or unload it from the console:

```text
ls /sd/models
load_model /sd/models/pulsevad_2.1k.espm
model
unload_model
```

`unload_model` restores the built-in Student model.

## Console commands

```text
info
selftest
golden
ls [path]
load_model <path.espm>
unload_model
model
threshold <0.0-1.0>
mic status
mic select <onboard|i2s>
mic gain <0|6|12|18|24|30|36|42>
speaker_vol <0-100>
test_wav <path.wav>
play_wav <path.wav>
record_mic <path.wav> <seconds>
vad_infer_mic <seconds>
wifi <setup|status|connect|disconnect|forget|token>
```

PulseVAD expects mono 16 kHz PCM16 audio and processes 3,200-sample
(200 ms) windows. `test_wav` processes complete windows and reports trailing
audio shorter than one window as ignored; it does not pad partial audio.

The onboard ES8311 microphone is selected by default unless another input was
saved. `mic status` reports the selected input and gain. `mic select onboard`
and `mic select i2s` change and save the input in NVS. `mic gain` accepts
0-42 dB in 6 dB steps, defaults to 36 dB, and also saves its setting in NVS.

`speaker_vol <0-100>` controls ES8311 playback volume. It defaults to 50%,
persists in NVS, and is applied by `play_wav`.

## Board interfaces

- ESP32-P4 SDMMC slot: CLK 43, CMD 44, D0-D3 39-42; powered by P4 LDO channel 4.
- ESP32-C6 WLAN co-processor: ESP-Hosted SDIO, configured in `sdkconfig.defaults`.
- Onboard ES8311 codec: I2C 7/8 and I2S0 9-13.
- External INMP441 microphone: I2S1 BCLK 20, WS 21, DIN 22.
- Interactive console: UART0, typically 115200 baud.
