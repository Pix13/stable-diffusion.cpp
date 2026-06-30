#include "runtime.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <regex>
#include <sstream>

#include "common/common.h"
#include "common/log.h"

namespace fs = std::filesystem;

static std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static bool is_supported_model_ext(const fs::path& p) {
    auto ext = lower_ascii(p.extension().string());
    return ext == ".gguf" || ext == ".pt" || ext == ".pth" || ext == ".safetensors";
}

static const std::string k_base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const std::vector<uint8_t>& bytes) {
    std::string ret;
    int val  = 0;
    int valb = -6;
    for (uint8_t c : bytes) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            ret.push_back(k_base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) {
        ret.push_back(k_base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    }
    while (ret.size() % 4) {
        ret.push_back('=');
    }
    return ret;
}

std::string normalize_output_format(std::string output_format) {
    std::transform(output_format.begin(), output_format.end(), output_format.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return output_format;
}

std::vector<std::string> supported_img_output_formats(bool allow_webp) {
    std::vector<std::string> formats = {"png", "jpeg"};
#ifdef SD_USE_WEBP
    if (allow_webp) {
        formats.push_back("webp");
    }
#else
    (void)allow_webp;
#endif
    return formats;
}

std::vector<std::string> supported_vid_output_formats() {
    std::vector<std::string> formats;
#ifdef SD_USE_WEBM
    formats.push_back("webm");
#endif
#ifdef SD_USE_WEBP
    formats.push_back("webp");
#endif
    formats.push_back("avi");
    return formats;
}

static std::string valid_vid_output_formats_message() {
    const std::vector<std::string> formats = supported_vid_output_formats();

    std::string message = "invalid output_format, must be one of [";
    for (size_t i = 0; i < formats.size(); ++i) {
        if (i > 0) {
            message += ", ";
        }
        message += formats[i];
    }
    message += "]";
    return message;
}

bool assign_output_options(ImgGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           bool allow_webp,
                           std::string& error_message) {
    request.output_format      = normalize_output_format(std::move(output_format));
    request.output_compression = std::clamp(output_compression, 0, 100);

    const std::vector<std::string> valid_formats = supported_img_output_formats(allow_webp);
    const bool valid_format                      = std::find(valid_formats.begin(),
                                                             valid_formats.end(),
                                                             request.output_format) != valid_formats.end();
    if (!valid_format) {
        error_message = "invalid output_format, must be one of [";
        for (size_t i = 0; i < valid_formats.size(); ++i) {
            if (i > 0) {
                error_message += ", ";
            }
            error_message += valid_formats[i];
        }
        error_message += "]";
        return false;
    }

    return true;
}

bool assign_output_options(VidGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           std::string& error_message) {
    request.output_format      = normalize_output_format(std::move(output_format));
    request.output_compression = std::clamp(output_compression, 0, 100);

    if (request.output_format == "avi") {
        return true;
    }

    if (request.output_format == "webm") {
#ifdef SD_USE_WEBM
        return true;
#else
        error_message = valid_vid_output_formats_message();
        return false;
#endif
    }

    if (request.output_format == "webp") {
#ifdef SD_USE_WEBP
        return true;
#else
        error_message = valid_vid_output_formats_message();
        return false;
#endif
    }

    error_message = valid_vid_output_formats_message();
    return false;
}

std::string video_mime_type(const std::string& output_format) {
    if (output_format == "webm") {
        return "video/webm";
    }
    if (output_format == "webp") {
        return "image/webp";
    }
    return "video/x-msvideo";
}

bool runtime_supports_generation_mode(const ServerRuntime& runtime, SDMode mode) {
    if (mode == VID_GEN) {
        return sd_ctx_supports_video_generation(runtime.sd_ctx);
    }
    if (mode == IMG_GEN) {
        return sd_ctx_supports_image_generation(runtime.sd_ctx);
    }
    return true;
}

std::string unsupported_generation_mode_error(SDMode mode) {
    if (mode == VID_GEN) {
        return "loaded model does not support vid_gen";
    }
    if (mode == IMG_GEN) {
        return "loaded model does not support img_gen";
    }
    return "loaded model does not support requested mode";
}

ArgOptions SDSvrParams::get_options() {
    ArgOptions options;

    options.string_options = {
        {"-l", "--listen-ip", "server listen ip (default: 127.0.0.1)", 0, &listen_ip},
        {"", "--serve-html-path", "path to HTML file to serve at root (optional)", 0, &serve_html_path},
    };

    options.int_options = {
        {"", "--listen-port", "server listen port (default: 1234)", &listen_port},
    };

    options.bool_options = {
        {"-v", "--verbose", "print extra info", true, &verbose},
        {"", "--color", "colors the logging tags according to level", true, &color},
    };

    auto on_help_arg = [&](int, const char**, int, bool& valid) {
        normal_exit = true;
        valid       = true;
        return -1;
    };

    auto on_load_lora = [&](int argc, const char** argv, int index, bool& valid) {
        if (index + 1 >= argc) {
            LOG_ERROR("error: --load-lora requires a value");
            valid = false;
            return -1;
        }
        std::string spec = argv[index + 1];
        // Trim leading/trailing whitespace
        size_t start = spec.find_first_not_of(" \t\r\n");
        size_t end   = spec.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) {
            LOG_ERROR("error: --load-lora: empty specification");
            valid = false;
            return -1;
        }
        spec = spec.substr(start, end - start + 1);

        // Split on semicolons
        std::vector<std::string> entries;
        std::stringstream ss(spec);
        std::string token;
        while (std::getline(ss, token, ';')) {
            // Trim whitespace
            size_t s = token.find_first_not_of(" \t\r\n");
            size_t e = token.find_last_not_of(" \t\r\n");
            if (s == std::string::npos) {
                continue; // skip empty entries
            }
            token = token.substr(s, e - s + 1);
            if (!token.empty()) {
                entries.push_back(token);
            }
        }

        for (const auto& entry : entries) {
            // Split on the last colon
            auto sep = entry.rfind(':');
            if (sep == std::string::npos) {
                LOG_ERROR("error: --load-lora: missing ':' separator in '%s'", entry.c_str());
                valid = false;
                return -1;
            }
            std::string path = entry.substr(0, sep);
            std::string strength_str = entry.substr(sep + 1);
            // Trim
            size_t ps = path.find_first_not_of(" \t\r\n");
            size_t pe = path.find_last_not_of(" \t\r\n");
            if (ps == std::string::npos) {
                LOG_ERROR("error: --load-lora: empty LoRA path in '%s'", entry.c_str());
                valid = false;
                return -1;
            }
            path = path.substr(ps, pe - ps + 1);
            size_t ss2 = strength_str.find_first_not_of(" \t\r\n");
            size_t se2 = strength_str.find_last_not_of(" \t\r\n");
            if (ss2 == std::string::npos) {
                LOG_ERROR("error: --load-lora: empty strength in '%s'", entry.c_str());
                valid = false;
                return -1;
            }
            strength_str = strength_str.substr(ss2, se2 - ss2 + 1);

            float strength;
            try {
                strength = std::stof(strength_str);
                if (!std::isfinite(strength)) {
                    LOG_ERROR("error: --load-lora: invalid strength '%s' in '%s'", strength_str.c_str(), entry.c_str());
                    valid = false;
                    return -1;
                }
            } catch (...) {
                LOG_ERROR("error: --load-lora: invalid strength '%s' in '%s'", strength_str.c_str(), entry.c_str());
                valid = false;
                return -1;
            }

            default_loras.push_back({path, strength});
        }
        return 1; // skip next arg
    };

    options.manual_options = {
        {"-h", "--help", "show this help message and exit", on_help_arg},
        {"", "--load-lora", "Default LoRA adapters to apply to server requests when the request payload does not provide LoRAs. Syntax: --load-lora path:strength[;path:strength...]", on_load_lora},
    };
    return options;
}

bool SDSvrParams::validate() {
    if (listen_ip.empty()) {
        LOG_ERROR("error: the following arguments are required: listen_ip");
        return false;
    }

    if (listen_port < 0 || listen_port > 65535) {
        LOG_ERROR("error: listen_port should be in the range [0, 65535]");
        return false;
    }

    if (!serve_html_path.empty() && !fs::exists(serve_html_path)) {
        LOG_ERROR("error: serve_html_path file does not exist: %s", serve_html_path.c_str());
        return false;
    }
    return true;
}

bool SDSvrParams::resolve_and_validate() {
    if (!validate()) {
        return false;
    }
    return true;
}

std::string SDSvrParams::to_string() const {
    std::ostringstream oss;
    oss << "SDSvrParams {\n"
        << "  listen_ip: " << listen_ip << ",\n"
        << "  listen_port: \"" << listen_port << "\",\n"
        << "  serve_html_path: \"" << serve_html_path << "\",\n";
    if (!default_loras.empty()) {
        for (size_t i = 0; i < default_loras.size(); ++i) {
            oss << "  default_lora[" << i << "]: path='" << default_loras[i].path
                << "', strength=" << default_loras[i].strength << "\n";
        }
    }
    oss << "}";
    return oss.str();
}

void refresh_lora_cache(ServerRuntime& rt) {
    std::vector<LoraEntry> new_cache;

    fs::path lora_dir = rt.ctx_params->lora_model_dir;
    if (fs::exists(lora_dir) && fs::is_directory(lora_dir)) {
        for (auto& entry : fs::recursive_directory_iterator(lora_dir, fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const fs::path& p = entry.path();
            if (!is_supported_model_ext(p)) {
                continue;
            }

            LoraEntry lora_entry;
            lora_entry.name     = p.stem().u8string();
            lora_entry.fullpath = p.u8string();
            std::string rel     = p.lexically_relative(lora_dir).u8string();
            std::replace(rel.begin(), rel.end(), '\\', '/');
            lora_entry.path = rel;

            new_cache.push_back(std::move(lora_entry));
        }
    }

    std::sort(new_cache.begin(), new_cache.end(), [](const LoraEntry& a, const LoraEntry& b) {
        return a.path < b.path;
    });

    {
        std::lock_guard<std::mutex> lock(*rt.lora_mutex);
        *rt.lora_cache = std::move(new_cache);
    }
}

std::string get_lora_full_path(ServerRuntime& rt, const std::string& path) {
    std::lock_guard<std::mutex> lock(*rt.lora_mutex);
    auto it = std::find_if(rt.lora_cache->begin(), rt.lora_cache->end(),
                           [&](const LoraEntry& entry) { return entry.path == path; });
    return it != rt.lora_cache->end() ? it->fullpath : "";
}

void refresh_upscaler_cache(ServerRuntime& rt) {
    std::vector<UpscalerEntry> new_cache;

    fs::path upscaler_dir = rt.ctx_params->hires_upscalers_dir;
    if (fs::exists(upscaler_dir) && fs::is_directory(upscaler_dir)) {
        for (auto& entry : fs::directory_iterator(upscaler_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const fs::path& p = entry.path();
            if (!is_supported_model_ext(p)) {
                continue;
            }

            UpscalerEntry upscaler_entry;
            upscaler_entry.name       = p.stem().u8string();
            upscaler_entry.fullpath   = fs::absolute(p).lexically_normal().u8string();
            upscaler_entry.model_name = "ESRGAN_4x";
            upscaler_entry.path       = p.filename().u8string();

            new_cache.push_back(std::move(upscaler_entry));
        }
    }

    std::sort(new_cache.begin(), new_cache.end(), [](const UpscalerEntry& a, const UpscalerEntry& b) {
        return a.name < b.name;
    });

    {
        std::lock_guard<std::mutex> lock(*rt.upscaler_mutex);
        *rt.upscaler_cache = std::move(new_cache);
    }
}

int64_t unix_timestamp_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void apply_default_loras(ServerRuntime& rt, SDGenerationParams& gen_params, bool request_provided_loras) {
    if (rt.svr_params->default_loras.empty()) {
        return;
    }
    if (request_provided_loras) {
        return;
    }
    for (const auto& lora : rt.svr_params->default_loras) {
        std::string fullpath = fs::path(lora.path).is_absolute()
                                    ? lora.path
                                    : get_lora_full_path(rt, lora.path);
        if (fullpath.empty()) {
            fullpath = lora.path;
        }
        gen_params.lora_map[fullpath] += lora.strength;
    }
}
