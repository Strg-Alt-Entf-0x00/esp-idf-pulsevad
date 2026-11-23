# Changelog

User-visible changes to PulseVAD are recorded here. Development experiments
and historical measurements are not release notes.

## [Unreleased]

### Added

- ESP-IDF C component for fixed-window mono audio VAD on ESP32-P4.
- Built-in Student model, configurable speech threshold and temporal
  hysteresis, and optional model-file loading.
- ESP32-P4 WLAN console example for microphone input, WAV inference, SD-card
  storage, and authenticated file transfer.

### Changed

- Optional model files use the ESPM v3 container format.
- Audio capture, resampling, and filesystem mounting remain application
  responsibilities.
