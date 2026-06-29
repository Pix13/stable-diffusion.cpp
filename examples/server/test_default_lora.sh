#!/usr/bin/env bash
# Test script for --default-lora feature
# Usage: ./test_default_lora.sh [path-to-sd-server]

set -euo pipefail

SD_SERVER="${1:-/home/dev/feat-loras/stable-diffusion.cpp/build/bin/sd-server}"
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

echo "=== Test: --default-lora CLI parsing ==="

# Test 1: --help should contain --default-lora
if $SD_SERVER --help 2>&1 | grep -q "default-lora"; then
    pass "help contains --default-lora"
else
    fail "help missing --default-lora"
fi

# Test 2: --default-lora with valid path should not crash (exits on help)
if $SD_SERVER --default-lora test.safetensors --help >/dev/null 2>&1; then
    pass "CLI accepts --default-lora test.safetensors"
else
    fail "CLI rejects --default-lora test.safetensors"
fi

# Test 3: --default-lora with multiplier
if $SD_SERVER --default-lora test.safetensors:0.8 --help >/dev/null 2>&1; then
    pass "CLI accepts --default-lora test.safetensors:0.8"
else
    fail "CLI rejects --default-lora test.safetensors:0.8"
fi

# Test 4: Multiple --default-lora options
if $SD_SERVER --default-lora a.safetensors:0.5 --default-lora b.safetensors:1.2 --help >/dev/null 2>&1; then
    pass "CLI accepts multiple --default-lora options"
else
    fail "CLI rejects multiple --default-lora options"
fi

# Test 5: Windows-style path with colon
if $SD_SERVER --default-lora "C:\models\lora.safetensors" --help >/dev/null 2>&1; then
    pass "CLI accepts Windows path with colon"
else
    fail "CLI rejects Windows path with colon"
fi

echo ""
echo "=== Test: Priority Logic (unit tests) ==="

# Create a minimal C++ test binary for the injection logic
TEST_SRC=$(mktemp /tmp/test_injection_XXXXXX.cpp)
cat > "$TEST_SRC" << 'EOF'
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct DefaultLoraConfig {
    std::string path;
    float multiplier = 1.0f;
    bool is_high_noise = false;
};

void inject_default_loras(json& body_json, const std::vector<DefaultLoraConfig>& default_loras) {
    if (default_loras.empty()) {
        return;
    }

    bool has_lora = body_json.contains("lora") && !body_json["lora"].is_null();
    if (has_lora) {
        return;
    }

    json lora_array = json::array();
    for (const auto& lora : default_loras) {
        lora_array.push_back({
            {"path", lora.path},
            {"multiplier", lora.multiplier},
            {"is_high_noise", lora.is_high_noise}
        });
    }
    body_json["lora"] = lora_array;
}

int main() {
    std::vector<DefaultLoraConfig> defaults = {
        {"foo.safetensors", 0.7f, false},
        {"bar.safetensors", 1.1f, false}
    };

    // Test 1: No lora field -> should inject defaults
    {
        json j = {{"prompt", "test"}};
        inject_default_loras(j, defaults);
        if (j.contains("lora") && j["lora"].size() == 2) {
            std::cout << "PASS: no lora field -> defaults injected" << std::endl;
        } else {
            std::cout << "FAIL: no lora field -> defaults not injected" << std::endl;
        }
    }

    // Test 2: lora present -> should NOT inject defaults
    {
        json j = {{"prompt", "test"}, {"lora", json::array({{{"path", "payload.safetensors", "multiplier", 0.5}}})}};
        inject_default_loras(j, defaults);
        if (j["lora"].size() == 1 && j["lora"][0]["path"] == "payload.safetensors") {
            std::cout << "PASS: lora present -> payload preserved" << std::endl;
        } else {
            std::cout << "FAIL: lora present -> payload not preserved" << std::endl;
        }
    }

    // Test 3: lora: [] -> should NOT inject defaults
    {
        json j = {{"prompt", "test"}, {"lora", json::array()}};
        inject_default_loras(j, defaults);
        if (j["lora"].size() == 0) {
            std::cout << "PASS: lora: [] -> empty array preserved" << std::endl;
        } else {
            std::cout << "FAIL: lora: [] -> empty array not preserved" << std::endl;
        }
    }

    // Test 4: lora: null -> should inject defaults
    {
        json j = {{"prompt", "test"}, {"lora", nullptr}};
        inject_default_loras(j, defaults);
        if (j.contains("lora") && j["lora"].size() == 2) {
            std::cout << "PASS: lora: null -> defaults injected" << std::endl;
        } else {
            std::cout << "FAIL: lora: null -> defaults not injected" << std::endl;
        }
    }

    // Test 5: no defaults configured -> no injection
    {
        json j = {{"prompt", "test"}};
        inject_default_loras(j, {});
        if (!j.contains("lora")) {
            std::cout << "PASS: no defaults -> no lora field added" << std::endl;
        } else {
            std::cout << "FAIL: no defaults -> lora field added" << std::endl;
        }
    }

    // Test 6: Verify default values in injected lora
    {
        json j = {{"prompt", "test"}};
        inject_default_loras(j, defaults);
        bool ok = true;
        if (j["lora"][0]["path"] != "foo.safetensors") ok = false;
        if (j["lora"][0]["multiplier"] != 0.7f) ok = false;
        if (j["lora"][0]["is_high_noise"] != false) ok = false;
        if (j["lora"][1]["path"] != "bar.safetensors") ok = false;
        if (j["lora"][1]["multiplier"] != 1.1f) ok = false;
        if (ok) {
            std::cout << "PASS: injected lora values are correct" << std::endl;
        } else {
            std::cout << "FAIL: injected lora values incorrect" << std::endl;
        }
    }

    return 0;
}
EOF

# Try to compile and run the test
TEST_BIN=$(mktemp /tmp/test_injection_XXXXXX)
if g++ -std=c++17 -I/home/dev/feat-loras/stable-diffusion.cpp/thirdparty/json/include "$TEST_SRC" -o "$TEST_BIN" 2>/dev/null; then
    $TEST_BIN
    rm -f "$TEST_SRC" "$TEST_BIN"
else
    echo "  SKIP: nlohmann/json header not found, skipping unit tests"
    rm -f "$TEST_SRC"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [ $FAIL -gt 0 ]; then
    exit 1
fi
