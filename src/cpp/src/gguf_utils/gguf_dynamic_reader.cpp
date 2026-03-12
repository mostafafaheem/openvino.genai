// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

#include "gguf_utils/gguf_dynamic_reader.hpp"

#include "utils.hpp"

#include "llama.h"
#include "utils.h"

GGUFReaderV2::GGUFReaderV2(const std::string& model_path)
    : m_model_path(model_path) {}

void GGUFReaderV2::initialize_backend_load_model() {
    llama_backend_init();

    auto model_params = llama_model_default_params();

    m_llama_model = llama_model_load_from_file(m_model_path.c_str(), model_params);
    if (!m_llama_model) {
        throw std::runtime_error(
            "GGUFReaderV2: Failed to load model from " + m_model_path);
    }
}

void GGUFReaderV2::trigger_ov_model_creation() {
    auto ctx_params = llama_context_default_params();
    ctx_params.n_threads = -1;
    ctx_params.n_threads_batch = -1;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    m_llama_ctx = llama_init_from_model(m_llama_model, ctx_params);
    if (!m_llama_ctx) {
        throw std::runtime_error(
            "GGUFReaderV2: Failed to create llama context");
    }
    ov_set_extract_mode(true);
    const llama_vocab* vocab = llama_model_get_vocab(m_llama_model);
    llama_token dummy_token = llama_vocab_bos(vocab);
    llama_batch batch = llama_batch_get_one(&dummy_token, 1);
    int rc = llama_decode(m_llama_ctx, batch);
    ov_set_extract_mode(false);
}

std::shared_ptr<ov::Model> GGUFReaderV2::extract_ov_model() {
    auto model = ov_get_extracted_model();
    if (!model) {
        throw std::runtime_error(
            "GGUFReaderV2: No OpenVINO model was extracted during graph build. ");
    }
    return model;
}

std::shared_ptr<ov::Model> GGUFReaderV2::convert() {
    initialize_backend_load_model();
    trigger_ov_model_creation();
    return extract_ov_model();
}

std::shared_ptr<ov::Model> create_from_gguf_v2(
    const std::string& model_path,
    const bool enable_save_ov_model) {

    GGUFReaderV2 reader(model_path);
    auto model = reader.convert();

    if (enable_save_ov_model) {
        std::filesystem::path gguf_model_path(model_path);
        std::filesystem::path save_path =
            gguf_model_path.parent_path() / "openvino_model_v2.xml";
        ov::genai::utils::save_openvino_model(model, save_path.string(), true);
    }
    return model;
}
