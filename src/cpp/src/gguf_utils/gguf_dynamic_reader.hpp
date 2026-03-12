// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#include "openvino/openvino.hpp"

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;

/**
 * @brief GGUFReaderV2 - Dynamic GGML graph-based GGUF reader.
 *
 * Instead of manually reconstructing model architectures layer-by-layer (V1),
 * this reader leverages llama.cpp to:
 *   1. Parse the GGUF file and load weights
 *   2. Build a GGML computation graph via llama.cpp's internal graph builder
 *   3. Translate the GGML graph into an OpenVINO ov::Model using the existing
 *      ov::frontend::ggml::FrontEnd from the llama.cpp OpenVINO backend
 *   4. Remap I/O names and shapes to match GenAI pipeline expectations
 *
 * This approach automatically supports all architectures that llama.cpp supports,
 * without requiring manual C++ implementation for each new topology.
 */
class GGUFReaderV2 {
public:
    explicit GGUFReaderV2(const std::string& model_path);
    // ~GGUFReaderV2();

    GGUFReaderV2(const GGUFReaderV2&) = delete;
    GGUFReaderV2& operator=(const GGUFReaderV2&) = delete;

    std::shared_ptr<ov::Model> convert();

private:
    void initialize_backend_load_model();

    void trigger_ov_model_creation();

    std::shared_ptr<ov::Model> extract_ov_model();

    void remap_io_names(std::shared_ptr<ov::Model>& model);

    std::string m_model_path;
    llama_model* m_llama_model = nullptr;
    llama_context* m_llama_ctx = nullptr;
};

/**
 * @brief Top-level entry point for GGUFReaderV2, matching the V1 create_from_gguf() signature.
 *
 * @param model_path Path to the .gguf model file
 * @param enable_save_ov_model If true, serialize the resulting ov::Model to disk
 * @return std::shared_ptr<ov::Model> The translated OpenVINO model
 */
std::shared_ptr<ov::Model> create_from_gguf_v2(
    const std::string& model_path,
    const bool enable_save_ov_model);
