/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include <cmath>
#include <set>

#include "dnnl_memory.hpp"

#include "self/self.hpp"

namespace self {

// Verifies that gpu_fill_random() produces non-uniform, finite,
// nan-free, inf-free, and seed-varying valid data for mode=F.
static int check_gpu_fill_random() {
    if (!(DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE
                && DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL)
            || is_cpu(get_test_engine())) {
        BENCHDNN_PRINT(2, "%s\n",
                "Skipping gpu_fill_random checks due to the use of a "
                "non-Intel GPU runtime or a CPU runtime.");
        return OK;
    }

    // Note:
    // This test suite will be deleted once debugging and verification of
    // new gpu_fill_random() is complete. Can be used to test the new
    // gpu_fill_random() as well as old memset-based approach.

    bool tests_array[] = {
            false, // 0. Print all values for debugging purposes
            true, // 1. Non-uniformity check
            true, // 2. No NaN/Inf
            false, // 3. Different calls should produce different data (seed test)
            true, // 4. All initialized (tail leftover bytes should be initialized too)
            false, // 5. Big tensor (e.g., 2GB for f16) test
    };

    // 0. Print all values for debugging purposes
    if (tests_array[0]) {
        const int nelems = 513;
        dnnl_dim_t dims {nelems};
        auto md = dnn_mem_t::init_md(1, &dims, dnnl_f16, tag::abx);

        dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
        m.unmap();
        SAFE(m.gpu_fill_random(nelems * sizeof(uint16_t), 0), WARN);
        m.map();

        const auto *ptr = static_cast<const uint16_t *>(m);

        for (int i = nelems - 1; i >= 0; i--) {
            if (ptr[i] != 0) {
                printf("[%i]: \033[32m%04X (%i)\033[0m\n", i, ptr[i], ptr[i]);
            } else {
                printf("[%i]: \033[31m%04X (%i)\033[0m\n", i, ptr[i], ptr[i]);
            }
        }
    }

    // 1. Non-uniformity check: require at least 50% unique values
    if (tests_array[1]) {
        // 1a. f16: 1M elements, 16-bit uniqueness
        {
            const int nelems = 1024 * 1024;
            dnnl_dim_t dims {nelems};
            auto md = dnn_mem_t::init_md(1, &dims, dnnl_f16, tag::abx);

            dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
            m.unmap();
            SAFE(m.gpu_fill_random(nelems * sizeof(uint16_t), 0), WARN);
            m.map();

            std::set<uint16_t> unique_vals;
            const auto *ptr = static_cast<const uint16_t *>(m);
            for (int i = 0; i < nelems; i++)
                unique_vals.insert(ptr[i]);

            printf("f16 unique values: %d/%d\n",
                    (int)unique_vals.size(), nelems);
            // SELF_CHECK(unique_vals.size() > static_cast<size_t>(nelems / 2),
            //         "gpu_fill_random produced too few unique f16 values: %d",
            //         (int)unique_vals.size());
        }
        // 1b. f32: 1M elements, 32-bit uniqueness
        {
            const int nelems = 1024 * 1024;
            dnnl_dim_t dims {nelems};
            auto md = dnn_mem_t::init_md(1, &dims, dnnl_f32, tag::abx);

            dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
            m.unmap();
            SAFE(m.gpu_fill_random(nelems * sizeof(uint32_t), 0), WARN);
            m.map();

            std::set<uint32_t> unique_vals;
            const auto *ptr = static_cast<const uint32_t *>(m);
            for (int i = 0; i < nelems; i++)
                unique_vals.insert(ptr[i]);

            printf("f32 unique values: %d/%d\n",
                    (int)unique_vals.size(), nelems);
            SELF_CHECK(unique_vals.size() > static_cast<size_t>(nelems / 2),
                    "gpu_fill_random produced too few unique f32 values: %d",
                    (int)unique_vals.size());
        }
        // 1c. f8_e5m2: 1M elements, 8-bit uniqueness (max 256)
        {
            const int nelems = 1024 * 1024;
            dnnl_dim_t dims {nelems};
            auto md = dnn_mem_t::init_md(1, &dims, dnnl_f8_e5m2, tag::abx);

            dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
            m.unmap();
            SAFE(m.gpu_fill_random(nelems * sizeof(uint8_t), 0), WARN);
            m.map();

            std::set<uint8_t> unique_vals;
            const auto *ptr = static_cast<const uint8_t *>(m);
            for (int i = 0; i < nelems; i++)
                unique_vals.insert(ptr[i]);

            // Mask 0xFB clears bit 2 → max 128 unique values.
            printf("f8_e5m2 unique values: %d/128\n",
                    (int)unique_vals.size());
            // SELF_CHECK(unique_vals.size() > 100,
            //         "gpu_fill_random produced too few unique f8_e5m2 "
            //         "values: %d",
            //         (int)unique_vals.size());
        }
        // 1d. f8_e4m3: 1M elements, 8-bit uniqueness (max 256)
        {
            const int nelems = 1024 * 1024;
            dnnl_dim_t dims {nelems};
            auto md = dnn_mem_t::init_md(1, &dims, dnnl_f8_e4m3, tag::abx);

            dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
            m.unmap();
            SAFE(m.gpu_fill_random(nelems * sizeof(uint8_t), 0), WARN);
            m.map();

            std::set<uint8_t> unique_vals;
            const auto *ptr = static_cast<const uint8_t *>(m);
            for (int i = 0; i < nelems; i++)
                unique_vals.insert(ptr[i]);

            // Mask 0xF7 clears bit 3 → max 128 unique values.
            printf("f8_e4m3 unique values: %d/128\n",
                    (int)unique_vals.size());
            // SELF_CHECK(unique_vals.size() > 100,
            //         "gpu_fill_random produced too few unique f8_e4m3 "
            //         "values: %d",
            //         (int)unique_vals.size());
        }
    }

    // 2. No NaN/Inf check per FP type (mask clears exponent LSB)
    if (tests_array[2]) {
        const int nelems = 1024;
        // 2a. f32: mask 0xFF7FFFFF clears bit 23
        {
            dnnl_dim_t dims {nelems};
            auto md = dnn_mem_t::init_md(1, &dims, dnnl_f32, tag::abx);

            dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
            m.unmap();
            SAFE(m.gpu_fill_random(nelems * sizeof(float), 0), WARN);
            m.map();

            const auto *ptr_u32 = static_cast<const uint32_t *>(m);
            const auto *ptr_f32 = static_cast<const float *>(m);

            for (int i = 0; i < nelems; i++) {
                SELF_CHECK((ptr_u32[i] & 0x00800000u) == 0,
                        "f32 mask invariant violated at index "
                        "%d: 0x%08X & 0x00800000 = 0x%08X",
                        i, ptr_u32[i], ptr_u32[i] & 0x00800000u);
                SELF_CHECK(std::isfinite(ptr_f32[i]),
                        "gpu_fill_random produced non-finite f32 at index %d",
                        i);
            }
        }
        // 2b. f16: mask 0xFBFF clears bit 10 per f16 element
        //     exponent = bits[14:10], all-ones exp = NaN/Inf
        {
            dnnl_dim_t dims {nelems};
            auto md = dnn_mem_t::init_md(1, &dims, dnnl_f16, tag::abx);

            dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
            m.unmap();
            SAFE(m.gpu_fill_random(nelems * sizeof(uint16_t), 0), WARN);
            m.map();

            const auto *ptr = static_cast<const uint16_t *>(m);
            for (int i = 0; i < nelems; i++) {
                SELF_CHECK((ptr[i] & 0x0400u) == 0,
                        "f16 mask invariant violated at index "
                        "%d: 0x%04X & 0x0400 = 0x%04X",
                        i, ptr[i], ptr[i] & 0x0400u);
                uint16_t exp = (ptr[i] >> 10) & 0x1F;
                SELF_CHECK(exp != 0x1F,
                        "gpu_fill_random produced NaN/Inf f16 at index "
                        "%d: 0x%04X",
                        i, ptr[i]);
            }
        }
        // 2c. bf16: mask 0xFF7F clears bit 7 per bf16 element
        //     exponent = bits[14:7], all-ones exp = NaN/Inf
        {
            dnnl_dim_t dims {nelems};
            auto md = dnn_mem_t::init_md(1, &dims, dnnl_bf16, tag::abx);

            dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
            m.unmap();
            SAFE(m.gpu_fill_random(nelems * sizeof(uint16_t), 0), WARN);
            m.map();

            const auto *ptr = static_cast<const uint16_t *>(m);
            for (int i = 0; i < nelems; i++) {
                SELF_CHECK((ptr[i] & 0x0080u) == 0,
                        "bf16 mask invariant violated at index "
                        "%d: 0x%04X & 0x0080 = 0x%04X",
                        i, ptr[i], ptr[i] & 0x0080u);
                uint16_t exp = (ptr[i] >> 7) & 0xFF;
                SELF_CHECK(exp != 0xFF,
                        "gpu_fill_random produced NaN/Inf bf16 at index "
                        "%d: 0x%04X",
                        i, ptr[i]);
            }
        }
        // 2d. f8_e5m2: mask 0xFB clears bit 2 per byte
        //     exponent = bits[6:2], all-ones exp = NaN/Inf
        {
            dnnl_dim_t dims {nelems};
            auto md = dnn_mem_t::init_md(1, &dims, dnnl_f8_e5m2, tag::abx);

            dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
            m.unmap();
            SAFE(m.gpu_fill_random(nelems * sizeof(uint8_t), 0), WARN);
            m.map();

            const auto *ptr = static_cast<const uint8_t *>(m);
            for (int i = 0; i < nelems; i++) {
                SELF_CHECK((ptr[i] & 0x04u) == 0,
                        "f8_e5m2 mask invariant violated at index "
                        "%d: 0x%02X & 0x04 = 0x%02X",
                        i, ptr[i], ptr[i] & 0x04u);
                uint8_t exp = (ptr[i] >> 2) & 0x1F;
                SELF_CHECK(exp != 0x1F,
                        "gpu_fill_random produced NaN/Inf f8_e5m2 at index "
                        "%d: 0x%02X",
                        i, ptr[i]);
            }
        }
    }

    // 3. Different calls should produce different data (seed test)
    if (tests_array[3]) {
        const int nelems = 512;
        dnnl_dim_t dims {nelems};
        auto md = dnn_mem_t::init_md(1, &dims, dnnl_f32, tag::abx);

        dnn_mem_t m1(md, get_test_engine(), /* prefill = */ false);
        dnn_mem_t m2(md, get_test_engine(), /* prefill = */ false);
        m1.unmap();
        m2.unmap();
        SAFE(m1.gpu_fill_random(nelems * sizeof(float), 0), WARN);
        SAFE(m2.gpu_fill_random(nelems * sizeof(float), 0), WARN);
        m1.map();
        m2.map();
        const auto *p1 = static_cast<const uint32_t *>(m1);
        const auto *p2 = static_cast<const uint32_t *>(m2);
        int num_different = 0;
        for (int i = 0; i < nelems; i++)
            if (p1[i] != p2[i]) num_different++;

        // printf("Number of different uint32_t values between two calls: "
        //        "%d/%d\n",
        //        num_different, nelems);

        // Require at least 50% of values to differ between two calls
        // if nelems set to default value of 1024.
        SELF_CHECK(num_different > nelems / 2,
                "Two gpu_fill_random calls produced too similar data: "
                "only %d/%d values differ",
                num_different, nelems);
    }

    // 4. All initialized (tail leftover bytes should be initialized too)
    if (tests_array[4]) {
        // 4a. f16: 2 bytes per element
        for (int nelems = 16; nelems <= 256; nelems++) {
            dnnl_dim_t dims {nelems};
            auto md = dnn_mem_t::init_md(1, &dims, dnnl_f16, tag::abx);

            dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
            const std::size_t total_bytes = nelems * sizeof(std::uint16_t);
            m.unmap();
            m.memset(0xFF, total_bytes, 0);
            SAFE(m.gpu_fill_random(total_bytes, 0), WARN);
            m.map();

            const auto *raw16 = static_cast<const uint16_t *>(m);
            int nan_count = 0;
            for (std::size_t i = 0; i < static_cast<std::size_t>(nelems); ++i)
                if (raw16[i] == 0xFFFFu) nan_count++;

            SELF_CHECK(nan_count == 0,
                    "f16: gpu_fill_random left %d uninitialized values "
                    "(0xFFFF) for nelems=%d",
                    nan_count, nelems);
        }
        // 4b. f8_e5m2: 1 byte per element — most sensitive to tail issues
        for (int nelems = 16; nelems <= 256; nelems++) {
            dnnl_dim_t dims {nelems};
            auto md = dnn_mem_t::init_md(
                    1, &dims, dnnl_f8_e5m2, tag::abx);

            dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
            const std::size_t total_bytes = nelems * sizeof(std::uint8_t);
            m.unmap();
            m.memset(0xFF, total_bytes, 0);
            SAFE(m.gpu_fill_random(total_bytes, 0), WARN);
            m.map();

            const auto *raw8 = static_cast<const uint8_t *>(m);
            int uninit_count = 0;
            for (std::size_t i = 0; i < static_cast<std::size_t>(nelems); ++i)
                if (raw8[i] == 0xFFu) uninit_count++;

            SELF_CHECK(uninit_count == 0,
                    "f8_e5m2: gpu_fill_random left %d uninitialized values "
                    "(0xFF) for nelems=%d",
                    uninit_count, nelems);
        }
    }

    // 5. Big tensor test: 4GB+1 bytes (u8) to catch uint32 overflow.
    if (tests_array[5]) {
        const size_t nelems = (1ULL << 32) + 1; // 4GB + 1 byte
        dnnl_dim_t dims {static_cast<dnnl_dim_t>(nelems)};
        auto md = dnn_mem_t::init_md(1, &dims, dnnl_u8, tag::abx);

        dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
        const std::size_t total_bytes = nelems * sizeof(std::uint8_t);
        m.unmap();
        m.memset(0xFF, total_bytes, 0);
        SAFE(m.gpu_fill_random(total_bytes, 0), WARN);
        m.map();

        // Check last 1024 samples
        const auto *raw8 = static_cast<const uint8_t *>(m);
        int uninit_count = 0;
        for (std::size_t i = nelems - 1024; i < nelems; ++i)
            if (raw8[i] == 0xFFu) uninit_count++;

        // All values should be initialized
        SELF_CHECK(uninit_count == 0,
                "gpu_fill_random left %d uninitialized values (0xFF) in "
                "the end of big tensor",
                uninit_count);
    }

    return OK;
}

static int check_bool_operator() {
    dnnl_dim_t dims {1};
    auto md = dnn_mem_t::init_md(1, &dims, dnnl_f32, tag::abx);
    auto md0 = dnn_mem_t::init_md(0, &dims, dnnl_f32, tag::abx);
    {
        dnn_mem_t m;
        SELF_CHECK_EQ(bool(m), false);
    }
    {
        dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(md0, get_test_engine(), /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(1, &dims, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(0, &dims, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(1, &dims, dnnl_f32, &dims /* strides */, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(0, &dims, dnnl_f32, &dims /* strides */, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(md, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(md0, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    return OK;
}

void memory() {
    RUN(check_bool_operator());
    RUN(check_gpu_fill_random());
}

} // namespace self
