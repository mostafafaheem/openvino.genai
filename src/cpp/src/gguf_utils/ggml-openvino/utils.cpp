#include "utils.hpp"
#include "ggml-impl.h"
#include "ggml-openvino-extra.hpp"
#include "ggml-decoder.hpp"
#include "ggml.h"
#include "openvino/frontend.hpp"
#include "openvino/input_model.hpp"

#include <memory>
#include <openvino/openvino.hpp>
#include <string>

// Thread-local export target for graph translation
thread_local std::shared_ptr<ov::Model>* g_export_target = nullptr;

void ggml_backend_openvino_set_export_target(std::shared_ptr<ov::Model>* target) {
    g_export_target = target;
}

bool is_naive(ggml_cgraph * cgraph) {
    constexpr int naive_graph_size_threshold = 20;
    int count = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    return count < naive_graph_size_threshold;
}

enum ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    if (!g_export_target) {
        // If no export target is set, we bypass execution since this backend 
        // has been repurposed strictly for graph translation.
        return GGML_STATUS_SUCCESS;
    }

    try {
        if (getenv("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        bool naive = is_naive(cgraph);
        if (naive) {
            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
            auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
            auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
            auto model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
            if (getenv("GGML_OPENVINO_DUMP_IR")) {
                ov::serialize(model, "IR_naive.xml");
            }
            *g_export_target = model;
            return GGML_STATUS_SUCCESS;
        }

        const auto is_static = ggml_openvino_is_npu();
        bool stateful = true;
        
        ModelParams m_params;
        ComputeParams c_params;
        std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

        auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);
        auto ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static, stateful);
        
        auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
        auto model = ov::frontend::ggml::FrontEnd::convert(input_model);
        ggml_decoder->clear_model_weights();

        if (getenv("GGML_OPENVINO_DUMP_IR")) {
            char timestamped_filename[64];
            auto timestamp = (long long) ggml_time_us();
            snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%lld.xml", timestamp);
            ov::serialize(model, timestamped_filename);
        }

        *g_export_target = model;
        return GGML_STATUS_SUCCESS;
        
    } catch (const ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        return GGML_STATUS_FAILED;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        return GGML_STATUS_FAILED;
    } catch (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        return GGML_STATUS_FAILED;
    }
}

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << std::endl;
    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        if (name.find("self_kq_mask") == std::string::npos) {
            std::cout << *(tensor.data<float>()) << std::endl;
        } else {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            auto * data = tensor.data<float>();
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    float val = data[i * cols + j];
                    if (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } else {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << std::endl;
            }
        }

        break;
    }
    case ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << std::endl;
        break;
    case ov::element::i32:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    case ov::element::i64:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    default:
        break;
    }
}

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, const void * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << std::endl;

    auto print_float_stats = [](const std::string & type_name, size_t size, auto get_value) {
        if (size == 0) {
            return;
        }

        float first = get_value(0);
        float min = first;
        float max = first;
        double sum = first;

        for (size_t i = 1; i < size; ++i) {
            float v = get_value(i);
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
            sum += v;
        }
        double mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << std::endl;
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << std::endl;
    };

    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        const float * data = tensor.data<float>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { return data[i]; });
        break;
    }
    case ov::element::f16: {
        const ov::float16 * data = tensor.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { return static_cast<float>(data[i]); });
        break;
    }
    default:
        break;
    }
}
