/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <vector>

#include "utils/parallel.hpp"

#include "sdpa/sdpa.hpp"

namespace sdpa {

// Reference SDPA implementation:
//   score = Q * K^T
//   score = score * scale  (or / scale if invert_scale)
//   score = score + mask   (if attention mask is present)
//   score = causal_mask(score)  (if causal mask)
//   prob = softmax(score)
//   out = prob * V
//
// All computation is done in f32.
void compute_ref(
        const prb_t *prb, dir_t dir, const args_t &args, dnnl_primitive_t) {
    const dnn_mem_t &q_m = args.find(DNNL_ARG_SRC_0);
    const dnn_mem_t &k_m = args.find(DNNL_ARG_SRC_1);
    const dnn_mem_t &v_m = args.find(DNNL_ARG_SRC_2);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);

    const bool has_mask = prb->with_mask();
    const bool has_causal = prb->with_causal_mask();
    const bool has_scale = prb->with_scale();

    const dnn_mem_t *mask_m = nullptr;
    const dnn_mem_t *scale_m = nullptr;
    if (has_mask) mask_m = &args.find(DNNL_ARG_SHIFT);
    if (has_scale) scale_m = &args.find(DNNL_ARG_SCALE);

    const int64_t MB = prb->mb;
    const int64_t Q = prb->n_queries;
    const int64_t K = prb->n_keys;
    const int64_t H = prb->head_size;
    const int64_t V = prb->n_values;
    const int nd = prb->ndims;

    // GQA/MQA: K/V may have fewer heads than Q. Compute the mapping.
    // For ndims >= 3, the heads dimension is at index ndims-3.
    const int64_t q_heads = (nd >= 3) ? prb->q_dims()[nd - 3] : 1;
    const int64_t kv_heads = (nd >= 3 && prb->kv_head_number > 0)
            ? prb->kv_head_number
            : q_heads;

    // Map a Q batch index to the corresponding K/V batch index.
    // When kv_heads < q_heads, multiple Q heads share one KV head.
    const auto kv_batch_idx = [&](int64_t mb_idx) -> int64_t {
        if (kv_heads == q_heads) return mb_idx;
        const int64_t outer = mb_idx / q_heads;
        const int64_t q_head = mb_idx % q_heads;
        const int64_t kv_head = q_head * kv_heads / q_heads;
        return outer * kv_heads + kv_head;
    };

    // Compute default scale = 1/sqrt(head_size).
    float scale_val = 1.0f / std::sqrt(static_cast<float>(H));
    if (has_scale) {
        float s = scale_m->get_f32_elem(0);
        scale_val = prb->invert_scale() ? 1.0f / s : s;
    }

    // Flat offset helpers assuming plain abx (row-major) layout.
    const auto q_off = [&](int64_t mb_idx, int64_t q_idx, int64_t h_idx) {
        return (mb_idx * Q + q_idx) * H + h_idx;
    };
    const auto k_off = [&](int64_t kv_mb, int64_t h_idx, int64_t k_idx) {
        return (kv_mb * H + h_idx) * K + k_idx;
    };
    const auto v_off = [&](int64_t kv_mb, int64_t k_idx, int64_t v_idx) {
        return (kv_mb * K + k_idx) * V + v_idx;
    };
    const auto dst_off = [&](int64_t mb_idx, int64_t q_idx, int64_t v_idx) {
        return (mb_idx * Q + q_idx) * V + v_idx;
    };
    const auto msk_off = [&](int64_t mb_idx, int64_t q_idx, int64_t k_idx) {
        return (mb_idx * Q + q_idx) * K + k_idx;
    };

    // Determine whether a position (q_idx, k_idx) is masked out by the causal mask
    // in the (query, key) index plane:
    //   - MASK_CAUSAL_TOP_LEFT: standard layout where future keys (k_idx > q_idx)
    //     are masked, i.e., the top-right triangle above the main diagonal.
    //   - "Bottom-right" layout: the causal band is shifted by (K - Q) so that
    //     future keys still lie above the effective diagonal; positions with
    //     k_idx > q_idx + (K - Q) are masked (only k_idx <= q_idx + K - Q are visible).
    const auto is_causal_masked
            = [&](int64_t q_idx, int64_t k_idx) -> bool {
        if (prb->mask_type == MASK_CAUSAL_TOP_LEFT) return k_idx > q_idx;
        // Bottom-right: visible iff k <= q + K - Q.
        return k_idx > q_idx + (K - Q);
    };

    benchdnn_parallel_nd(MB, Q, [&](int64_t mb_idx, int64_t q_idx) {
        const int64_t kv_mb = kv_batch_idx(mb_idx);

        // Step 1: score[k] = Q[q,:] * K[:,k].
        std::vector<float> score(K);
        for (int64_t k_idx = 0; k_idx < K; k_idx++) {
            float acc = 0.0f;
            for (int64_t h_idx = 0; h_idx < H; h_idx++) {
                acc += q_m.get_f32_elem(q_off(mb_idx, q_idx, h_idx))
                        * k_m.get_f32_elem(k_off(kv_mb, h_idx, k_idx));
            }
            score[k_idx] = acc;
        }

        // Step 2: Scale.
        for (auto &s : score)
            s *= scale_val;

        // Step 3: Add attention mask if present.
        if (has_mask) {
            for (int64_t k_idx = 0; k_idx < K; k_idx++)
                score[k_idx]
                        += mask_m->get_f32_elem(msk_off(mb_idx, q_idx, k_idx));
        }

        // Step 4: Apply causal mask.
        if (has_causal) {
            for (int64_t k_idx = 0; k_idx < K; k_idx++)
                if (is_causal_masked(q_idx, k_idx))
                    score[k_idx] = -std::numeric_limits<float>::infinity();
        }

        // Step 5: Softmax over K dimension.
        {
            float max_val = *std::max_element(score.begin(), score.end());
            float sum = 0.0f;
            for (auto &s : score) {
                s = std::exp(s - max_val);
                sum += s;
            }
            const float inv_sum = (sum != 0.0f) ? 1.0f / sum : 0.0f;
            for (auto &s : score)
                s *= inv_sum;
        }

        // Step 6: output = prob * V.
        for (int64_t v_idx = 0; v_idx < V; v_idx++) {
            float acc = 0.0f;
            for (int64_t k_idx = 0; k_idx < K; k_idx++)
                acc += score[k_idx]
                        * v_m.get_f32_elem(v_off(kv_mb, k_idx, v_idx));
            dst_m.set_f32_elem(dst_off(mb_idx, q_idx, v_idx), acc);
        }
    });
}

} // namespace sdpa
