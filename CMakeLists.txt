idf_component_register(
    SRCS
        "src/esp_pulsevad.c"
        "src/loaders/pulsevad_espm_loader_core.c"
        "src/loaders/pulsevad_espm_loader_vfs.c"
        "src/pulsevad_frontend.c"
        "src/pulsevad_nn.c"
    INCLUDE_DIRS
        "include"
    PRIV_INCLUDE_DIRS
        "src"
    REQUIRES
        esp-dsp
        esp_timer
        esp-idf-model-manager
)

# ESP-IDF derives the component name from this directory. Update application
# REQUIRES entries whenever the checkout directory is renamed.

# Aggressive floating-point/compiler options. Validate numerical behavior and
# performance separately for each target before adding it to the support matrix.
target_compile_options(${COMPONENT_LIB} PRIVATE 
    -Ofast                    # Enable -O3 + fast floating-point math
    -funroll-loops            # Force loop unrolling for convolution & matrix ops
    -fno-strict-aliasing      # Allow aggressive pointer optimizations
    -falign-functions=32      # 32-byte function alignment for ESP32-P4 cache (32KB L1)
    -falign-loops=16          # 16-byte loop alignment for rapid instruction fetch
)

# Keep non-finite ESPM payload validation reliable despite the inference
# component's fast-math optimization flags.
set_source_files_properties(src/loaders/pulsevad_espm_loader_core.c PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math"
)
