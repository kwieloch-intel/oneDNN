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
// Each work item processes 16 bytes (4 x uint32) using Philox vec4.
__kernel void fill_random(__global uint *buf, uint seed, ulong byte_count) {
    ulong start = get_global_id(0) * 16;
    if (start >= byte_count) return;

    uint b = (uint)(start >> 2);
    uint4 rnd = philox_4x32_vec4(b, b ^ seed) & (uint4)(0xEEEEEEEEu);

    if (start + 16 <= byte_count) {
        vstore4(rnd, get_global_id(0), buf);
    } else {
        __global uchar *p = (__global uchar *)buf;
        uint r[4] = {rnd.s0, rnd.s1, rnd.s2, rnd.s3};
        for (ulong i = 0; i < byte_count - start; ++i)
            p[start + i] = (uchar)(r[i / 4] >> ((i % 4) * 8));
    }
}
