#pragma once

// TQ2_0 dot-product kernel selection.
//
//   GGML_CPU_TQ2_KERNEL=old   the shift-and-mask unpack (3 USHR + 3 AND per 16 code bytes)
//   GGML_CPU_TQ2_KERNEL=new   the mask-only unpack (3 AND + 1 USHR), the default
//
// Both forms feed the same raw codes {0,1,2} to the same dot products in the same order and
// subtract the same bsums total, so the integer sum entering the float epilogue is identical
// and the two settings differ in instruction count only. The switch exists so that a timing
// claim can be made against the old kernel in the same binary, and as a kill switch.

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_tq2_kernel_init(void);   // re-read the environment (tests call this between arms)
bool ggml_tq2_kernel_new(void);    // true = mask-only unpack

#ifdef __cplusplus
}
#endif
