/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cstring>
#include <vector>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "gated_mlp/gated_mlp.hpp"

namespace gated_mlp {

// Reference GatedMLP: generates gold data by composing existing oneDNN
// primitives (matmul, eltwise) instead of reimplementing from scratch.
//
// Pipeline:
//   up     = matmul(src, W_up)            -> [MB, OC]
//   gate   = matmul(src, W_gate)          -> [MB, OC]
//   gate   = eltwise(gate, activation)    -> [MB, OC]
//   gate   = gate * up                    -> [MB, OC]  (element-wise)
//   dst    = matmul(gate, W_down)         -> [MB, IC]
//
// All intermediate computation is done in f32 on the CPU engine.

namespace {

// Execute a matmul primitive on CPU: dst = src x wei.
void exec_matmul(dnnl_engine_t eng, dnnl_stream_t strm, const dnn_mem_t &src,
        const dnn_mem_t &wei, const dnn_mem_t &dst) {
    dnnl_primitive_desc_t pd {};
    DNN_SAFE_V(dnnl_matmul_primitive_desc_create(
            &pd, eng, src.md_, wei.md_, nullptr, dst.md_, nullptr));
    auto pd_w = make_benchdnn_dnnl_wrapper(pd);

    dnnl_primitive_t prim {};
    DNN_SAFE_V(dnnl_primitive_create(&prim, pd));
    auto prim_w = make_benchdnn_dnnl_wrapper(prim);

    dnnl_exec_arg_t args[] = {
            {DNNL_ARG_SRC, src.m_},
            {DNNL_ARG_WEIGHTS, wei.m_},
            {DNNL_ARG_DST, dst.m_},
    };
    DNN_SAFE_V(dnnl_primitive_execute(prim, strm, 3, args));
    DNN_SAFE_V(dnnl_stream_wait(strm));
}

// Execute in-place eltwise forward on CPU.
void exec_eltwise(dnnl_engine_t eng, dnnl_stream_t strm, dnn_mem_t &mem,
        dnnl_alg_kind_t alg) {
    // Swish uses alpha=1.0 by default. GELU variants ignore alpha/beta.
    float alpha = (alg == dnnl_eltwise_swish) ? 1.0f : 0.0f;
    float beta = 0.0f;

    dnnl_primitive_desc_t pd {};
    DNN_SAFE_V(dnnl_eltwise_forward_primitive_desc_create(&pd, eng,
            dnnl_forward_inference, alg, mem.md_, mem.md_, alpha, beta,
            nullptr));
    auto pd_w = make_benchdnn_dnnl_wrapper(pd);

    dnnl_primitive_t prim {};
    DNN_SAFE_V(dnnl_primitive_create(&prim, pd));
    auto prim_w = make_benchdnn_dnnl_wrapper(prim);

    dnnl_exec_arg_t args[] = {
            {DNNL_ARG_SRC, mem.m_},
            {DNNL_ARG_DST, mem.m_},
    };
    DNN_SAFE_V(dnnl_primitive_execute(prim, strm, 2, args));
    DNN_SAFE_V(dnnl_stream_wait(strm));
}

// Execute in-place binary operation on CPU: lhs = lhs <alg> rhs.
void exec_binary(dnnl_engine_t eng, dnnl_stream_t strm, dnn_mem_t &lhs,
        const dnn_mem_t &rhs, dnnl_alg_kind_t alg) {
    dnnl_primitive_desc_t pd {};
    DNN_SAFE_V(dnnl_binary_primitive_desc_create(
            &pd, eng, alg, lhs.md_, rhs.md_, lhs.md_, nullptr));
    auto pd_w = make_benchdnn_dnnl_wrapper(pd);

    dnnl_primitive_t prim {};
    DNN_SAFE_V(dnnl_primitive_create(&prim, pd));
    auto prim_w = make_benchdnn_dnnl_wrapper(prim);

    dnnl_exec_arg_t args[] = {
            {DNNL_ARG_SRC_0, lhs.m_},
            {DNNL_ARG_SRC_1, rhs.m_},
            {DNNL_ARG_DST, lhs.m_},
    };
    DNN_SAFE_V(dnnl_primitive_execute(prim, strm, 3, args));
    DNN_SAFE_V(dnnl_stream_wait(strm));
}

// Create a 2-D f32 plain memory on `eng`.
dnn_mem_t make_2d(dnnl_engine_t eng, int64_t d0, int64_t d1) {
    dnnl_dims_t dims = {d0, d1};
    auto md = dnn_mem_t::init_md(2, dims, dnnl_f32, tag::abx);
    return dnn_mem_t(md, eng, /* prefill = */ false);
}

// Dequantize a 2D weight tensor in-place: w[k][n] = scale[idx] * (w[k][n] - zp[idx]).
// For weights shaped [K, N]:
//   mask=2 (PER_OC):   scale/zp indexed by n only.
//   mask=3 (PER_OCIC): scale/zp indexed by (k / group_k) * N + n.
void dequantize_2d(float *w, int64_t K, int64_t N, const dnn_mem_t &scales_m,
        const dnn_mem_t &zps_m, bool has_scale, bool has_zp, int scale_mask,
        int zp_mask, const std::vector<dnnl_dim_t> &scale_groups,
        const std::vector<dnnl_dim_t> &zp_groups) {
    if (!has_scale && !has_zp) return;

    // Determine K-group size for scales and zero-points.
    // mask bit 1 (1<<0) set means per-K; group_k subdivides K dimension.
    const int64_t scale_group_k = (scale_mask & 1)
            ? (!scale_groups.empty() ? scale_groups[0] : 1)
            : K;
    const int64_t zp_group_k
            = (zp_mask & 1) ? (!zp_groups.empty() ? zp_groups[0] : 1) : K;
    // N dimension for indexing into scale/zp arrays.
    const int64_t scale_N = (scale_mask & 2) ? N : 1;
    const int64_t zp_N = (zp_mask & 2) ? N : 1;

    for (int64_t k = 0; k < K; ++k) {
        for (int64_t n = 0; n < N; ++n) {
            const int64_t s_idx
                    = (k / scale_group_k) * scale_N + (scale_N > 1 ? n : 0);
            const int64_t z_idx = (k / zp_group_k) * zp_N + (zp_N > 1 ? n : 0);
            const float scale = has_scale ? scales_m.get_f32_elem(s_idx) : 1.f;
            const int zp = has_zp ? zps_m.get_elem(z_idx) : 0;
            w[k * N + n] = scale * (w[k * N + n] - zp);
        }
    }
}

} // anonymous namespace

void compute_ref(
        const prb_t *prb, dir_t dir, const args_t &args, dnnl_primitive_t) {
    const auto &eng = get_cpu_engine();
    dnnl_stream_t strm {};
    DNN_SAFE_V(dnnl_stream_create(&strm, eng, dnnl_stream_default_flags));

    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &w_gate_m = args.find(DNNL_ARG_WEIGHTS_GATE);
    const dnn_mem_t &w_up_m = args.find(DNNL_ARG_WEIGHTS_UP);
    const dnn_mem_t &w_down_m = args.find(DNNL_ARG_WEIGHTS_DOWN);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);

    const int64_t MB = prb->mb;
    const int64_t IC = prb->ic;
    const int64_t OC = prb->oc;

    // 2-D f32 memories for intermediate results.
    auto up_result = make_2d(eng, MB, OC);
    auto gate_result = make_2d(eng, MB, OC);

    // Dequantize weights if quantization attributes are set.
    const auto &attr = prb->attr;
    auto dequant_wei
            = [&](const dnn_mem_t &w, int64_t K, int64_t N, int wei_arg) {
                  const bool has_scale = !attr.scales.get(wei_arg).is_def();
                  const bool has_zp
                          = !attr.zero_points.get(wei_arg).is_def();
                  if (!has_scale && !has_zp) return dnn_mem_t();

                  const dnn_mem_t &sc
                          = args.find(DNNL_ARG_ATTR_SCALES | wei_arg);
                  const dnn_mem_t &zp
                          = args.find(DNNL_ARG_ATTR_ZERO_POINTS | wei_arg);
                  const int sc_mask = attr.scales.get_mask(
                          wei_arg, dnnl_undefined_primitive, 2 /*ndims*/);
                  const int zp_mask
                          = has_zp ? attr.zero_points.get_mask(
                                    wei_arg, dnnl_undefined_primitive, 2)
                                   : 0;
                  const auto &sc_groups = attr.scales.get(wei_arg).groups;
                  const auto &zp_groups
                          = has_zp ? attr.zero_points.get(wei_arg).groups
                                   : std::vector<dnnl_dim_t> {};
                  auto retn = make_2d(eng, K, N);
                  std::memcpy((float *)retn, (float *)w,
                          K * N * sizeof(float));
                  dequantize_2d((float *)retn, K, N, sc, zp, has_scale, has_zp,
                          sc_mask, zp_mask, sc_groups, zp_groups);
                  return retn;
              };

    auto w_gate_ref = dequant_wei(w_gate_m, IC, OC, DNNL_ARG_WEIGHTS_GATE);
    auto w_up_ref = dequant_wei(w_up_m, IC, OC, DNNL_ARG_WEIGHTS_UP);
    auto w_down_ref = dequant_wei(w_down_m, OC, IC, DNNL_ARG_WEIGHTS_DOWN);

    // Step 1: up_result = matmul(src, W_up).
    exec_matmul(eng, strm, src_m, (w_up_ref) ? w_up_ref : w_up_m, up_result);

    // Step 2: gate_result = matmul(src, W_gate).
    exec_matmul(eng, strm, src_m, (w_gate_ref) ? w_gate_ref : w_gate_m,
            gate_result);

    // Step 3: gate_result = activation(gate_result).
    exec_eltwise(eng, strm, gate_result, prb->activation);

    // Step 4: up_result = up_result * gate_result (element-wise).
    exec_binary(eng, strm, up_result, gate_result, dnnl_binary_mul);

    // Step 5: dst = matmul(up_result, W_down).
    exec_matmul(
            eng, strm, up_result, (w_down_ref) ? w_down_ref : w_down_m, dst_m);

    DNN_SAFE_V(dnnl_stream_destroy(strm));
}

} // namespace gated_mlp
