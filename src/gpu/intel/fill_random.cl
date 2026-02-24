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

#define DT_UNDEF 1
#include "gpu/intel/include/philox.h"

// Fills a buffer with pseudo-random uint32 values using the Philox RNG.
__kernel void fill_random(__global uint *buf, uint seed, uint byte_count) {
    uint gid = (uint)get_global_id(0);
    uint value = philox_4x32(gid, gid ^ seed) & 0xEEEEEEEEu;

    if (gid < (byte_count >> 2)) {
        buf[gid] = value;
    } else {
        __global uchar *p = (__global uchar *)(buf + gid);
        uint tail = byte_count & 3;
        if (tail > 0) p[0] = (uchar)value;
        if (tail > 1) p[1] = (uchar)(value >> 8);
        if (tail > 2) p[2] = (uchar)(value >> 16);
    }
}

#ifdef SIMD_WIDTH
// Fills a buffer with pseudo-random uint32 values using the Philox RNG,
// can use SIMD instruction and block processing for better performance.
__kernel void fill_random_vec(__global uint *buf, uint seed, uint byte_count) {
    const uint id = (uint)(get_global_id(0) * BLOCK_SIZE);
    const uint elem_count = byte_count >> 2;
    SIMD_VECTOR vec;

#pragma unroll BLOCK_SIZE / SIMD_WIDTH
    for (int i = 0; i < BLOCK_SIZE; i += SIMD_WIDTH) {
        uint offset = id + i;

        // Generate a vector of 4 random values using Philox.
#pragma unroll SIMD_WIDTH / 4
        for (int k = 0; k < SIMD_WIDTH; k += 4) {
            uint base = offset + k;
            uint4 rnd = philox_4x32_vec4(base, base ^ seed);
            vec[k] = rnd.s0;
            vec[k + 1] = rnd.s1;
            vec[k + 2] = rnd.s2;
            vec[k + 3] = rnd.s3;
        }
        vec &= (SIMD_VECTOR)(0xEEEEEEEEu);

        // Regular case: full SIMD (SIMD_WIDTH) vector store.
        if (offset + SIMD_WIDTH <= elem_count) {
            SIMD_STORE(vec, 0, buf + offset);
            continue;
        }

        // Tail: store samples that don't fit a full SIMD vector.
        int remaining = elem_count - offset;
        for (int k = 0; k < remaining; k++)
            buf[offset + k] = vec[k];

        // Tail: store bytes that don't fit a single 32-bit uint.
        uint tail = byte_count & 3;
        if (tail > 0) {
            __global uchar *p = (__global uchar *)(buf + offset + remaining);
            uint value = vec[remaining];
            p[0] = (uchar)value;
            if (tail > 1) p[1] = (uchar)(value >> 8);
            if (tail > 2) p[2] = (uchar)(value >> 16);
        }
        return;
    }
}
#endif
