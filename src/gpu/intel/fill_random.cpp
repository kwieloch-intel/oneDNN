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

#include <mutex>
#include <unordered_map>

#include "common/c_types_map.hpp"

#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/engine.hpp"
#include "gpu/intel/stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

// Set to true to use the reference kernel.
static constexpr bool use_ref_kernel = false;

// Vectorized kernel compile-time parameters.
static constexpr int simd_width = 16; // in {4, 8, 16}
static constexpr int block_size = 64; // k * simd_width, k in {1, 2, 4, ...}

inline size_t ceil_div(size_t x, size_t y) {
    return (x + y - 1) / y;
}

static compute::kernel_t get_cached_kernel(intel::engine_t *engine) {
    static std::unordered_map<engine_id_t, compute::kernel_t> cache;
    static std::mutex mutex;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(engine->engine_id());
    if (it != cache.end()) return it->second;

    compute::kernel_ctx_t ctx;
    ctx.define_int("BLOCK_SIZE", block_size);
    ctx.define_int("SIMD_WIDTH", simd_width);
    ctx.add_option("-DSIMD_VECTOR=uint" + std::to_string(simd_width));
    ctx.add_option("-DSIMD_STORE=vstore" + std::to_string(simd_width));

    std::vector<compute::kernel_t> kernels;
    const char *name = use_ref_kernel ? "fill_random" : "fill_random_vec";
    UNUSED_STATUS(engine->create_kernels(&kernels, {name}, ctx));
    return cache.emplace(engine->engine_id(), kernels[0]).first->second;
}

status_t fill_random(impl::stream_t *stream, size_t size,
        impl::memory_t *memory, int buffer_index, uint32_t seed) {
    if (size == 0) return status::success;

    auto *intel_stream = utils::downcast<intel::stream_t *>(stream);
    auto *intel_engine = utils::downcast<intel::engine_t *>(stream->engine());
    auto kernel = get_cached_kernel(intel_engine);

    const size_t bytes_per_item = use_ref_kernel ? 4 : block_size * 4;
    compute::nd_range_t nd_range({ceil_div(size, bytes_per_item), 1, 1});
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, *memory->memory_storage(buffer_index));
    arg_list.set(1, seed);
    arg_list.set(2, static_cast<uint64_t>(size));

    CHECK(kernel.parallel_for(*stream, nd_range, arg_list,
            intel_stream->ctx().get_deps(), intel_stream->ctx().get_deps()));
    return status::success;
}

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

extern "C" dnnl::impl::status_t DNNL_API dnnl_impl_gpu_fill_random(
        dnnl::impl::stream_t *stream, size_t size, dnnl::impl::memory_t *memory,
        int buffer_index, uint32_t seed) {
    return dnnl::impl::gpu::intel::fill_random(
            stream, size, memory, buffer_index, seed);
}
