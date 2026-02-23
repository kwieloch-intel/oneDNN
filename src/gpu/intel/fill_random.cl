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
// but can use SIMD instruction and block processing for better performance.
__kernel void fill_random_vec(__global uint *buf, uint seed, uint byte_count) {
    uint id = (uint)(get_global_id(0) * BLOCK_SIZE);
    SIMD_VECTOR vec;

    #pragma unroll BLOCK_SIZE / SIMD_WIDTH
    for (int i=0; i<BLOCK_SIZE; i+=SIMD_WIDTH) {
        // Generate a vector of random values using Philox.
        #pragma unroll SIMD_WIDTH
        for (int k=0; k<SIMD_WIDTH; k++){
            vec[k] = philox_4x32(id + i + k, (id + i + k) ^ seed) & 0xEEEEEEEEu;
        }

        // Check regular execution or tail case.
        if(id + i + SIMD_WIDTH > byte_count / 4){
            // Here we handle the tail, the last block is partial,
            // we can't use simd_store because the number of elements
            // is arbitrary and unknown.
            int remaining_elements = (byte_count / 4) - (id + i);
            for(int k=0; k<remaining_elements; k++){
                 buf[id + i + k] = vec[k];
            }

            // Handle bytes that don't fit into a full uint32 (4 bytes)
            int remaining_bytes = byte_count & 3;
            if (remaining_bytes > 0) {
                __global uchar *p = (__global uchar *)(buf + id + i + remaining_elements);
                uint value = vec[remaining_elements];
                if (remaining_bytes > 0) p[0] = (uchar)value;
                if (remaining_bytes > 1) p[1] = (uchar)(value >> 8);
                if (remaining_bytes > 2) p[2] = (uchar)(value >> 16);
            }
        }
        else{
            // Here we have a full block, we can use simd_store
            SIMD_STORE(vec, 0, buf + id + i);
        }
    }
}
#endif