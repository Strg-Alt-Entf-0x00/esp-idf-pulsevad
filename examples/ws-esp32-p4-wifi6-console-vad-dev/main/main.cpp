/**
 * @file main.cpp
 * @brief PulseVAD Console Application
 * 
 * Target: Waveshare ESP32-P4-WIFI6 Development Board
 * Features:
 * - ESPM model storage on SD card
 * - Authenticated WLAN file transfer
 * - Dual microphone support (ES8311 + INMP441)
 * - Interactive console interface
 */

#include <cstdio>
#include <cstring>
#include <new>
#include <errno.h>
#include <sys/stat.h>
#include "esp_log.h"
#include "esp_console.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "soc/soc_caps.h"
#include "nvs_flash.h"
#include "esp_pm.h"
#include "esp_private/esp_clk.h"

#include "app_config.h"
#include "audio_manager.h"
#include "cmd_vad_cli.h"
#include "esp_pulsevad.h"
#include "wifi_transfer.h"

static const char* TAG = "main";

// ============================================================================
// Banner & System Info
// ============================================================================

static void print_banner(void) {
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t total_sram = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    
    uint32_t cpu_mhz = esp_clk_cpu_freq() / 1000000;
    
    printf("\n");
    printf("  PULSEVAD CONSOLE\n");
    printf("  ESP32-P4 Edition\n");
    printf("  --------------------------------------------------\n");
    printf("  Hardware : ESP32-P4 rev 1.3 (2-Core @ %u MHz)\n", (unsigned)cpu_mhz);
    printf("  PSRAM    : %u / %u MB (Free/Total)\n",
           (unsigned)(free_psram / (1024*1024)),
           (unsigned)(total_psram / (1024*1024)));
    printf("  SRAM     : %u / %u KB (Free/Total)\n",
           (unsigned)(free_sram / 1024),
           (unsigned)(total_sram / 1024));
    printf("  Version  : v1.0.0\n\n");
}

// ============================================================================
// SD card mounting
// ============================================================================

#if defined(CONFIG_IDF_TARGET_ESP32P4) && \
    defined(CONFIG_ESP_HOSTED_HOST_TRANSPORT_BUS_SDIO) && \
    ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
static esp_err_t shared_sdmmc_host_init(void) { return ESP_OK; }
static esp_err_t shared_sdmmc_host_deinit(void) { return ESP_OK; }
#endif

static esp_err_t mount_sd_card(void)
{
#if SOC_SDMMC_HOST_SUPPORTED
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;
#if defined(CONFIG_IDF_TARGET_ESP32P4)
    host.slot = SDMMC_HOST_SLOT_0;
#endif
#if defined(CONFIG_IDF_TARGET_ESP32P4) && \
    defined(CONFIG_ESP_HOSTED_HOST_TRANSPORT_BUS_SDIO) && \
    ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
    host.init = shared_sdmmc_host_init;
    host.deinit = shared_sdmmc_host_deinit;
#endif

#if defined(CONFIG_IDF_TARGET_ESP32P4)
    sd_pwr_ctrl_ldo_config_t ldo_config = {.ldo_chan_id = 4};
    sd_pwr_ctrl_handle_t power_control = nullptr;
    const esp_err_t power_error =
        sd_pwr_ctrl_new_on_chip_ldo(&ldo_config, &power_control);
    if (power_error != ESP_OK) {
        return power_error;
    }
    host.pwr_ctrl_handle = power_control;
#endif

    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
#if defined(CONFIG_IDF_TARGET_ESP32P4)
    slot.clk = static_cast<gpio_num_t>(43);
    slot.cmd = static_cast<gpio_num_t>(44);
    slot.d0 = static_cast<gpio_num_t>(39);
    slot.d1 = static_cast<gpio_num_t>(40);
    slot.d2 = static_cast<gpio_num_t>(41);
    slot.d3 = static_cast<gpio_num_t>(42);
    slot.width = 4;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
#endif

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {};
    mount_config.format_if_mount_failed = false;
    mount_config.max_files = 10;
    mount_config.allocation_unit_size = 16 * 1024;
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
    mount_config.disk_status_check_enable = true;
    mount_config.use_one_fat = false;
#endif

    static sdmmc_card_t *card = nullptr;
    const esp_err_t error = esp_vfs_fat_sdmmc_mount(
        FS_MOUNT_POINT_SD, &host, &slot, &mount_config, &card);
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "SD-card mount failed: %s", esp_err_to_name(error));
#if defined(CONFIG_IDF_TARGET_ESP32P4)
        sd_pwr_ctrl_del_on_chip_ldo(power_control);
#endif
        return error;
    }

    sdmmc_card_print_info(stdout, card);
    const char *directories[] = {"/sd/models", "/sd/audio", "/sd/results"};
    for (const char *directory : directories) {
        if (mkdir(directory, 0775) != 0 && errno != EEXIST) {
            ESP_LOGE(TAG, "Cannot create %s: errno=%d", directory, errno);
            return ESP_FAIL;
        }
    }
    ESP_LOGI(TAG, "SD card mounted at %s", FS_MOUNT_POINT_SD);
    return ESP_OK;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

// ============================================================================
// Main Application
// ============================================================================

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Starting PulseVAD System");
    
    // ========================================
    // NVS Flash
    // ========================================
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // ========================================
    // Power Management
    // ========================================
#if CONFIG_PM_ENABLE
    esp_pm_config_t pm_config = {
        .max_freq_mhz = 360,
        .min_freq_mhz = 40,
        .light_sleep_enable = true
    };
    ESP_ERROR_CHECK(esp_pm_configure(&pm_config));
    ESP_LOGI(TAG, "Power management configured (max 360MHz, light sleep enabled)");
#endif
    
    // ========================================
    // SD card and WLAN transfer service
    // ========================================
    const esp_err_t sd_error = mount_sd_card();
    if (sd_error != ESP_OK) {
        ESP_LOGW(TAG, "SD card unavailable; model and file commands may fail: %s",
                 esp_err_to_name(sd_error));
    }

    const esp_err_t wifi_error = wifi_transfer_init();
    if (wifi_error != ESP_OK) {
        ESP_LOGW(TAG, "WLAN transfer initialization failed: %s", esp_err_to_name(wifi_error));
    }
    
    // ========================================
    // Console Setup (before audio to prevent blocking)
    // ========================================
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_config.prompt = CONSOLE_PROMPT;
    repl_config.max_cmdline_length = CONSOLE_MAX_COMMAND_LINE;
    repl_config.task_stack_size = 8192;
    
    cmd_vad_cli_register();
    wifi_transfer_register_commands();
    
    esp_console_dev_uart_config_t hw_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&hw_config, &repl_config, &repl));
    
    print_banner();
    
    // ========================================
    // Audio Manager (non-blocking init)
    // ========================================
    ret = audio_manager_init();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Audio initialization failed: %s", esp_err_to_name(ret));
        ESP_LOGW(TAG, "Audio features disabled; file-based VAD still available");
        printf("  ⚠ Audio hardware unavailable\n");
    } else {
        ESP_LOGI(TAG, "Audio hardware initialized");
        audio_manager_play_boot_sequence();
    }
    
    // ========================================
    // Status & Instructions
    // ========================================
    printf("  Type 'help' to see all commands.\n");
    if (sd_error == ESP_OK) {
        printf("  SD card : /sd (mounted)\n");
        printf("  Models  : /sd/models/*.espm\n");
        printf("  Audio   : /sd/audio/*.wav\n");
    } else {
        printf("  SD card : unavailable (insert card and restart)\n");
        printf("  Models  : unavailable\n");
    }
    printf("\n");
    
    ESP_LOGI(TAG, "Console ready");
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}
