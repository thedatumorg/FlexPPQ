#include <immintrin.h>
#include <limits>
#include <cstddef>

#ifndef PREFETCH_HINT
// 0=_MM_HINT_NTA, 1=_MM_HINT_T0, 2=_MM_HINT_T1, 3=_MM_HINT_T2
#define PREFETCH_HINT _MM_HINT_T0
#endif

static inline void prefetch_vec_head(const float* p, int dim_floats) {
    // 只 prefetch 头部几条 cache line：一般足够把流水线“点燃”
    // 64B cache line = 16 floats
    // 这里拉 4 条线（256B）你可以按 dim 调
    (void)dim_floats;
    _mm_prefetch(reinterpret_cast<const char*>(p + 0),  PREFETCH_HINT);
    _mm_prefetch(reinterpret_cast<const char*>(p + 16), PREFETCH_HINT);
    _mm_prefetch(reinterpret_cast<const char*>(p + 32), PREFETCH_HINT);
    _mm_prefetch(reinterpret_cast<const char*>(p + 48), PREFETCH_HINT);
}

