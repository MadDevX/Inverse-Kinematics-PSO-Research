#pragma once
#include "simplex.h"
#include <float.h>

#define CCD_INIT(ccd) \
    do { \
        (ccd)->first_dir = ccdFirstDirDefault; \
        (ccd)->support1 = NULL; \
        (ccd)->support2 = NULL; \
        (ccd)->center1  = NULL; \
        (ccd)->center2  = NULL; \
        \
        (ccd)->max_iterations = (unsigned long)-1; \
        (ccd)->epa_tolerance = CCD_REAL(0.0001); \
        (ccd)->mpr_tolerance = CCD_REAL(0.0001); \
        (ccd)->dist_tolerance = CCD_REAL(1E-6); \
    } while(0)

#define ZERO 0.0f
#define ONE 1.0f
#define COL_EPS FLT_EPSILON
#define F3ZERO make_float3(0.0f,0.0f,0.0f)


struct collisionData {
	unsigned long max_iterations; //!< Maximal number of iterations
	float epa_tolerance;
	float mpr_tolerance; //!< Boundary tolerance for MPR algorithm
	float dist_tolerance;
};

typedef struct collisionData collisionData_t;


void firstDir(const void *obj1, const void *obj2, float3 *dir);

void supportBox(const void *obj, const float3 *dir, float3 *vec);

void centerBox(const void *obj1, float3 *center);

int ccdGJKIntersect(const void *obj1, const void *obj2, const collisionData_t *ccd)
{
	simplex_t simplex;
	return __ccdGJK(obj1, obj2, ccd, &simplex) == 0;
}


static int __ccdGJK(const void *obj1, const void *obj2,
	const collisionData_t *ccd, simplex_t *simplex)
{
	unsigned long iterations;
	float3 dir; // direction vector
	support_t last; // last support point
	int do_simplex_res;

	// initialize simplex struct
	ccdSimplexInit(simplex);

	// get first direction
	firstDir(obj1, obj2, &dir);
	// get first support point
	__ccdSupport(obj1, obj2, &dir, ccd, &last);
	// and add this point to simplex as last one
	ccdSimplexAdd(simplex, &last);

	// set up direction vector to as (O - last) which is exactly -last
	ccdVec3Copy(&dir, &last.v);
	ccdVec3Scale(&dir, -ONE);

	// start iterations
	for (iterations = 0UL; iterations < ccd->max_iterations; ++iterations) {
		// obtain support point
		__ccdSupport(obj1, obj2, &dir, ccd, &last);

		// check if farthest point in Minkowski difference in direction dir
		// isn't somewhere before origin (the test on negative dot product)
		// - because if it is, objects are not intersecting at all.
		if (ccdVec3Dot(&last.v, &dir) < ZERO) {
			return -1; // intersection not found
		}

		// add last support vector to simplex
		ccdSimplexAdd(simplex, &last);

		// if doSimplex returns 1 if objects intersect, -1 if objects don't
		// intersect and 0 if algorithm should continue
		do_simplex_res = doSimplex(simplex, &dir);
		if (do_simplex_res == 1) {
			return 0; // intersection found
		}
		else if (do_simplex_res == -1) {
			return -1; // intersection not found
		}

		if (ccdIsZERO(ccdVec3Len2(&dir))) {
			return -1; // intersection not found
		}
	}

	// intersection wasn't found
	return -1;
}

void __ccdSupport(const void *obj1, const void *obj2,
	const float3 *_dir, const collisionData_t *ccd,
	support_t *supp)
{
	float3 dir;

	ccdVec3Copy(&dir, _dir);

	//wklejenie w v1 wyniku wywolania funkcji support1
	supportBox(obj1, &dir, &supp->v1);

	ccdVec3Scale(&dir, -ONE);
	//wklejenie w v2 wyniku wywolania funkcji support2
	supportBox(obj2, &dir, &supp->v2);

	//roznica minkowskiego
	ccdVec3Sub2(&supp->v, &supp->v1, &supp->v2);
}


#pragma region doSimplexi

static int doSimplex(simplex_t *simplex, float3 *dir)
{
	if (ccdSimplexSize(simplex) == 2) {
		// simplex contains segment only one segment
		return doSimplex2(simplex, dir);
	}
	else if (ccdSimplexSize(simplex) == 3) {
		// simplex contains triangle
		return doSimplex3(simplex, dir);
	}
	else { // ccdSimplexSize(simplex) == 4
	   // tetrahedron - this is the only shape which can encapsule origin
	   // so doSimplex4() also contains test on it
		return doSimplex4(simplex, dir);
	}
}

static int doSimplex2(simplex_t *simplex, float3 *dir)
{
	const support_t *A, *B;
	float3 AB, AO, tmp;
	float dot;

	// get last added as A
	A = ccdSimplexLast(simplex);
	// get the other point
	B = ccdSimplexPoint(simplex, 0);
	// compute AB oriented segment
	ccdVec3Sub2(&AB, &B->v, &A->v);
	// compute AO vector
	ccdVec3Copy(&AO, &A->v);
	ccdVec3Scale(&AO, -ONE);

	// dot product AB . AO
	dot = ccdVec3Dot(&AB, &AO);

	// check if origin doesn't lie on AB segment
	ccdVec3Cross(&tmp, &AB, &AO);
	if (ccdIsZERO(ccdVec3Len2(&tmp)) && dot > ZERO) {
		return 1;
	}

	// check if origin is in area where AB segment is
	if (ccdIsZERO(dot) || dot < ZERO) {
		// origin is in outside are of A
		ccdSimplexSet(simplex, 0, A);
		ccdSimplexSetSize(simplex, 1);
		ccdVec3Copy(dir, &AO);
	}
	else {
		// origin is in area where AB segment is

		// keep simplex untouched and set direction to
		// AB x AO x AB
		tripleCross(&AB, &AO, &AB, dir);
	}

	return 0;
}

static int doSimplex3(simplex_t *simplex, float3 *dir)
{
	const support_t *A, *B, *C;
	float3 AO, AB, AC, ABC, tmp;
	float dot, dist;

	// get last added as A
	A = ccdSimplexLast(simplex);
	// get the other points
	B = ccdSimplexPoint(simplex, 1);
	C = ccdSimplexPoint(simplex, 0);

	// check touching contact
	dist = ccdVec3PointTriDist2(F3ZERO, &A->v, &B->v, &C->v, NULL);
	if (ccdIsZERO(dist)) {
		return 1;
	}

	// check if triangle is really triangle (has area > 0)
	// if not simplex can't be expanded and thus no itersection is found
	if (ccdVec3Eq(&A->v, &B->v) || ccdVec3Eq(&A->v, &C->v)) {
		return -1;
	}

	// compute AO vector
	ccdVec3Copy(&AO, &A->v);
	ccdVec3Scale(&AO, -ONE);

	// compute AB and AC segments and ABC vector (perpendircular to triangle)
	ccdVec3Sub2(&AB, &B->v, &A->v);
	ccdVec3Sub2(&AC, &C->v, &A->v);
	ccdVec3Cross(&ABC, &AB, &AC);

	ccdVec3Cross(&tmp, &ABC, &AC);
	dot = ccdVec3Dot(&tmp, &AO);
	if (ccdIsZERO(dot) || dot > ZERO) {
		dot = ccdVec3Dot(&AC, &AO);
		if (ccdIsZERO(dot) || dot > ZERO) {
			// C is already in place
			ccdSimplexSet(simplex, 1, A);
			ccdSimplexSetSize(simplex, 2);
			tripleCross(&AC, &AO, &AC, dir);
		}
		else {
		ccd_do_simplex3_45:
			dot = ccdVec3Dot(&AB, &AO);
			if (ccdIsZERO(dot) || dot > ZERO) {
				ccdSimplexSet(simplex, 0, B);
				ccdSimplexSet(simplex, 1, A);
				ccdSimplexSetSize(simplex, 2);
				tripleCross(&AB, &AO, &AB, dir);
			}
			else {
				ccdSimplexSet(simplex, 0, A);
				ccdSimplexSetSize(simplex, 1);
				ccdVec3Copy(dir, &AO);
			}
		}
	}
	else {
		ccdVec3Cross(&tmp, &AB, &ABC);
		dot = ccdVec3Dot(&tmp, &AO);
		if (ccdIsZERO(dot) || dot > ZERO) {
			goto ccd_do_simplex3_45;
		}
		else {
			dot = ccdVec3Dot(&ABC, &AO);
			if (ccdIsZERO(dot) || dot > ZERO) {
				ccdVec3Copy(dir, &ABC);
			}
			else {
				support_t Ctmp;
				ccdSupportCopy(&Ctmp, C);
				ccdSimplexSet(simplex, 0, B);
				ccdSimplexSet(simplex, 1, &Ctmp);

				ccdVec3Copy(dir, &ABC);
				ccdVec3Scale(dir, -ONE);
			}
		}
	}

	return 0;
}

static int doSimplex4(simplex_t *simplex, float3 *dir)
{
	const support_t *A, *B, *C, *D;
	float3 AO, AB, AC, AD, ABC, ACD, ADB;
	int B_on_ACD, C_on_ADB, D_on_ABC;
	int AB_O, AC_O, AD_O;
	float dist;

	// get last added as A
	A = ccdSimplexLast(simplex);
	// get the other points
	B = ccdSimplexPoint(simplex, 2);
	C = ccdSimplexPoint(simplex, 1);
	D = ccdSimplexPoint(simplex, 0);

	// check if tetrahedron is really tetrahedron (has volume > 0)
	// if it is not simplex can't be expanded and thus no intersection is
	// found
	dist = ccdVec3PointTriDist2(&A->v, &B->v, &C->v, &D->v, NULL);
	if (ccdIsZERO(dist)) {
		return -1;
	}

	// check if origin lies on some of tetrahedron's face - if so objects
	// intersect
	dist = ccdVec3PointTriDist2(F3ZERO, &A->v, &B->v, &C->v, NULL);
	if (ccdIsZERO(dist))
		return 1;
	dist = ccdVec3PointTriDist2(F3ZERO, &A->v, &C->v, &D->v, NULL);
	if (ccdIsZERO(dist))
		return 1;
	dist = ccdVec3PointTriDist2(F3ZERO, &A->v, &B->v, &D->v, NULL);
	if (ccdIsZERO(dist))
		return 1;
	dist = ccdVec3PointTriDist2(F3ZERO, &B->v, &C->v, &D->v, NULL);
	if (ccdIsZERO(dist))
		return 1;

	// compute AO, AB, AC, AD segments and ABC, ACD, ADB normal vectors
	ccdVec3Copy(&AO, &A->v);
	ccdVec3Scale(&AO, -ONE);
	ccdVec3Sub2(&AB, &B->v, &A->v);
	ccdVec3Sub2(&AC, &C->v, &A->v);
	ccdVec3Sub2(&AD, &D->v, &A->v);
	ccdVec3Cross(&ABC, &AB, &AC);
	ccdVec3Cross(&ACD, &AC, &AD);
	ccdVec3Cross(&ADB, &AD, &AB);

	// side (positive or negative) of B, C, D relative to planes ACD, ADB
	// and ABC respectively
	B_on_ACD = ccdSign(ccdVec3Dot(&ACD, &AB));
	C_on_ADB = ccdSign(ccdVec3Dot(&ADB, &AC));
	D_on_ABC = ccdSign(ccdVec3Dot(&ABC, &AD));

	// whether origin is on same side of ACD, ADB, ABC as B, C, D
	// respectively
	AB_O = ccdSign(ccdVec3Dot(&ACD, &AO)) == B_on_ACD;
	AC_O = ccdSign(ccdVec3Dot(&ADB, &AO)) == C_on_ADB;
	AD_O = ccdSign(ccdVec3Dot(&ABC, &AO)) == D_on_ABC;

	if (AB_O && AC_O && AD_O) {
		// origin is in tetrahedron
		return 1;

		// rearrange simplex to triangle and call doSimplex3()
	}
	else if (!AB_O) {
		// B is farthest from the origin among all of the tetrahedron's
		// points, so remove it from the list and go on with the triangle
		// case

		// D and C are in place
		ccdSimplexSet(simplex, 2, A);
		ccdSimplexSetSize(simplex, 3);
	}
	else if (!AC_O) {
		// C is farthest
		ccdSimplexSet(simplex, 1, D);
		ccdSimplexSet(simplex, 0, B);
		ccdSimplexSet(simplex, 2, A);
		ccdSimplexSetSize(simplex, 3);
	}
	else { // (!AD_O)
		ccdSimplexSet(simplex, 0, C);
		ccdSimplexSet(simplex, 1, B);
		ccdSimplexSet(simplex, 2, A);
		ccdSimplexSetSize(simplex, 3);
	}

	return doSimplex3(simplex, dir);
}

static int doSimplex(simplex_t *simplex, float3 *dir)
{
	if (ccdSimplexSize(simplex) == 2) {
		// simplex contains segment only one segment
		return doSimplex2(simplex, dir);
	}
	else if (ccdSimplexSize(simplex) == 3) {
		// simplex contains triangle
		return doSimplex3(simplex, dir);
	}
	else { // ccdSimplexSize(simplex) == 4
	   // tetrahedron - this is the only shape which can encapsule origin
	   // so doSimplex4() also contains test on it
		return doSimplex4(simplex, dir);
	}
}




#pragma endregion

#pragma region  inlines
void ccdSimplexInit(simplex_t *s)
{
	s->last = -1;
}


float ccdVec3Len2(const float3 *v)
{
	return ccdVec3Dot(v, v);
}

int ccdIsZERO(float val)
{
	return CCD_FABS(val) < COL_EPS;
}

void ccdSimplexAdd(simplex_t *s, const support_t *v)
{
	// here is no check on boundaries in sake of speed
	++s->last;
	ccdSupportCopy(s->ps + s->last, v);
}


#pragma endregion

