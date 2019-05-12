#pragma once
#include "cuda_runtime.h"

struct support {
	float3 v;  //!< Support point in minkowski sum
	float3 v1; //!< Support point in obj1
	float3 v2; //!< Support point in obj2
};

typedef struct support support_t;

void SupportCopy(support_t *, const support_t *s);

inline void SupportCopy(support_t *d, const support_t *s)
{
	*d = *s;
}


