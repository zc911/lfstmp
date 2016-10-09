
#ifndef  __COMMON_H_
#define  __COMMON_H_
#include <xmmintrin.h>

#ifdef _MSC_VER /* visual c++ */
# define ALIGN16_BEG __declspec(align(16))
# define ALIGN16_END 
#else /* gcc or icc */
# define ALIGN16_BEG
# define ALIGN16_END __attribute__((aligned(16)))
#endif

typedef __m128 v4sf;  // vector of 4 float (sse1)

#ifdef USE_SSE2
# include <emmintrin.h>
typedef __m128i v4si; // vector of 4 int (sse2)
#else
typedef __m64 v2si;   // vector of 2 int (mmx)
#endif

/* declare some SSE constants -- why can't I figure a better way to do that? */
#define _PS_CONST(Name, Val)                                            \
  static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
  static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }

#include "constants.h"

#if defined (__MINGW32__)

/* the ugly part below: many versions of gcc used to be completely buggy with respect to some intrinsics
   The movehl_ps is fixed in mingw 3.4.5, but I found out that all the _mm_cmp* intrinsics were completely
   broken on my mingw gcc 3.4.5 ...

   Note that the bug on _mm_cmp* does occur only at -O0 optimization level
*/

inline __m128 my_movehl_ps(__m128 a, const __m128 b) {
	asm (
			"movhlps %2,%0\n\t"
			: "=x" (a)
			: "0" (a), "x"(b)
	    );
	return a;                                 }
#warning "redefined _mm_movehl_ps (see gcc bug 21179)"
#define _mm_movehl_ps my_movehl_ps

inline __m128 my_cmplt_ps(__m128 a, const __m128 b) {
	asm (
			"cmpltps %2,%0\n\t"
			: "=x" (a)
			: "0" (a), "x"(b)
	    );
	return a;               
                  }
inline __m128 my_cmpgt_ps(__m128 a, const __m128 b) {
	asm (
			"cmpnleps %2,%0\n\t"
			: "=x" (a)
			: "0" (a), "x"(b)
	    );
	return a;               
}
inline __m128 my_cmpeq_ps(__m128 a, const __m128 b) {
	asm (
			"cmpeqps %2,%0\n\t"
			: "=x" (a)
			: "0" (a), "x"(b)
	    );
	return a;               
}
#warning "redefined _mm_cmpxx_ps functions..."
#define _mm_cmplt_ps my_cmplt_ps
#define _mm_cmpgt_ps my_cmpgt_ps
#define _mm_cmpeq_ps my_cmpeq_ps
#endif

#ifndef USE_SSE2
typedef union xmm_mm_union {
  __m128 xmm;
  __m64 mm[2];
} xmm_mm_union;

#define COPY_XMM_TO_MM(xmm_, mm0_, mm1_) {          \
    xmm_mm_union u; u.xmm = xmm_;                   \
    mm0_ = u.mm[0];                                 \
    mm1_ = u.mm[1];                                 \
}

#define COPY_MM_TO_XMM(mm0_, mm1_, xmm_) {                         \
    xmm_mm_union u; u.mm[0]=mm0_; u.mm[1]=mm1_; xmm_ = u.xmm;      \
  }

#endif // USE_SSE2
















#endif  //__COMMON_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
