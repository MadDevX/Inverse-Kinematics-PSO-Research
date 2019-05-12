#pragma once
#include "cuda_runtime.h"

struct support {
	float3 v;  //!< Support point in minkowski sum
	float3 v1; //!< Support point in obj1
	float3 v2; //!< Support point in obj2
};

typedef struct support support_t;

void SupportCopy(support_t *, const support_t *s);

/**
	* Computes support point of obj1 and obj2 in direction dir.
	* Support point is returned via supp.
	*/
void __Support(const void *obj1, const void *obj2,
	const float3 *dir, const collisionData_t *,
	support_t *supp);

inline void SupportCopy(support_t *d, const support_t *s)
{
	*d = *s;
}


