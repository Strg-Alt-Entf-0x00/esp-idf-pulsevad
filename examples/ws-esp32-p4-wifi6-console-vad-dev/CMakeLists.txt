# PulseVAD Console Developer Example
cmake_minimum_required(VERSION 3.16)

# Add the PulseVAD component.
set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../.."
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(pulsevad_console_dev)
