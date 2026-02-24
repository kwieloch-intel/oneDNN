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

inline size_t ceil_div(size_t x, size_t y) {
    return (x + y - 1) / y;
}

//////////////////////////////////////////////////////////

static compute::kernel_t get_cached_kernel(intel::engine_t *engine) {
    static std::unordered_map<engine_id_t, compute::kernel_t> cache;
    static std::mutex mutex;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(engine->engine_id());
    if (it != cache.end()) return it->second;

    compute::kernel_ctx_t ctx;
    std::vector<compute::kernel_t> kernels;
    UNUSED_STATUS(engine->create_kernels(&kernels, {"fill_random"}, ctx));
    return cache.emplace(engine->engine_id(), kernels[0]).first->second;
}

status_t fill_random(impl::stream_t *stream, size_t size,
        impl::memory_t *memory, int buffer_index, uint32_t seed) {
    if (size == 0) return status::success;

    auto *intel_stream = utils::downcast<intel::stream_t *>(stream);
    auto *intel_engine = utils::downcast<intel::engine_t *>(stream->engine());

    auto kernel = get_cached_kernel(intel_engine);

    const size_t num_work_items = ceil_div(size, 4);
    printf("fill_random: size=%zu, size/4=%zu, num_work_items=%zu\n", size, size / 4, num_work_items);

    compute::range_t gws = {num_work_items, 1, 1};
    compute::nd_range_t nd_range(gws);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, *memory->memory_storage(buffer_index));
    arg_list.set(1, seed);
    arg_list.set(2, static_cast<uint32_t>(size));

    CHECK(kernel.parallel_for(*stream, nd_range, arg_list,
            intel_stream->ctx().get_deps(), intel_stream->ctx().get_deps()));

    return status::success;
}

//////////////////////////////////////////////////////////

struct kernel_cache_key_t {
    engine_id_t engine_id;
    int block_size;
    int simd_width;

    bool operator==(const kernel_cache_key_t &other) const {
        return engine_id == other.engine_id && block_size == other.block_size
                && simd_width == other.simd_width;
    }
};

struct kernel_cache_key_hash_t {
    size_t operator()(const kernel_cache_key_t &k) const {
        size_t h = std::hash<engine_id_t>()(k.engine_id);
        h ^= std::hash<int>()(k.block_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.simd_width) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static compute::kernel_t get_cached_kernel_vec(
        intel::engine_t *engine, int block_size, int simd_width) {
    static std::unordered_map<kernel_cache_key_t, compute::kernel_t,
            kernel_cache_key_hash_t>
            cache;
    static std::mutex mutex;

    kernel_cache_key_t key {engine->engine_id(), block_size, simd_width};

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    compute::kernel_ctx_t ctx;
    ctx.define_int("BLOCK_SIZE", block_size);
    ctx.define_int("SIMD_WIDTH", simd_width);
    ctx.add_option("-DSIMD_VECTOR=uint" + std::to_string(simd_width));
    ctx.add_option("-DSIMD_STORE=vstore" + std::to_string(simd_width));

    std::vector<compute::kernel_t> kernels;
    UNUSED_STATUS(engine->create_kernels(&kernels, {"fill_random_vec"}, ctx));
    return cache.emplace(key, kernels[0]).first->second;
}

status_t fill_random_vec(impl::stream_t *stream, size_t size,
        impl::memory_t *memory, int buffer_index, uint32_t seed) {
    if (size == 0) return status::success;

    auto *intel_stream = utils::downcast<intel::stream_t *>(stream);
    auto *intel_engine = utils::downcast<intel::engine_t *>(stream->engine());

    // Available SIMD_WIDTH = {4, 8, 16}
    // Available BLOCK_SIZE = k * SIMD_WIDTH, where k is a small positive integer (e.g., 1, 2, 4, 8).
    int block_size= 64, simd_width = 16;
    auto kernel = get_cached_kernel_vec(intel_engine, block_size, simd_width);

    const size_t num_work_items = ceil_div(size, block_size * 4);
    printf("fill_random_vec: size=%zu, size/4=%zu, num_work_items=%zu\n", size, size / 4, num_work_items);

    compute::range_t gws = {num_work_items, 1, 1};
    compute::nd_range_t nd_range(gws);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, *memory->memory_storage(buffer_index));
    arg_list.set(1, seed);
    arg_list.set(2, static_cast<uint32_t>(size));

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
    return dnnl::impl::gpu::intel::fill_random_vec(
            stream, size, memory, buffer_index, seed);
}
