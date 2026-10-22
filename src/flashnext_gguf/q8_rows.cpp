// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstring>
#include <cmath>

static float half_to_float(uint16_t h) {
    const int sign = (h >> 15) ? -1 : 1;
    const int exp = (h >> 10) & 31;
    const int frac = h & 1023;
    if (!exp) return sign * std::ldexp(float(frac), -24);
    if (exp == 31) return frac ? NAN : sign * INFINITY;
    return sign * std::ldexp(float(1024 + frac), exp - 25);
}
static uint16_t to_bf16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    if ((x & 0x7fffffffU) > 0x7f800000U) return uint16_t((x >> 16) | 0x0040);
    return uint16_t((x + 0x7fffU + ((x >> 16) & 1)) >> 16);
}

// Q8_0: each 32-value block stores an IEEE FP16 scale then 32 signed bytes.
// Validate every row before touching output. No mutable state; concurrent calls are safe.
extern "C" int q8_get_rows(const uint8_t *table, int64_t rows, int64_t width,
                          const int64_t *ids, int64_t count, void *output, int dtype) {
    if (rows < 0 || width <= 0 || width % 32 || count < 0 || (dtype != 0 && dtype != 1)) return -1;
    for (int64_t i = 0; i < count; ++i) if (ids[i] < 0 || ids[i] >= rows) return -2;
    const int64_t blocks = width / 32;
    for (int64_t i = 0; i < count; ++i) {
        const uint8_t *row = table + ids[i] * blocks * 34;
        for (int64_t b = 0; b < blocks; ++b) {
            uint16_t h; std::memcpy(&h, row + b * 34, 2);
            float scale = half_to_float(h);
            const int8_t *q = reinterpret_cast<const int8_t *>(row + b * 34 + 2);
            for (int64_t j = 0; j < 32; ++j) {
                const int64_t k = i * width + b * 32 + j;
                const float value = scale * float(q[j]);
                if (dtype == 0) static_cast<float *>(output)[k] = value;
                else static_cast<uint16_t *>(output)[k] = to_bf16(value);
            }
        }
    }
    return 0;
}
