#pragma once
#include "simplex.h"
#include <float.h>
#include "vector_operations.cuh"
#include <vector_types.h>
#define CCD_INIT(ccd) \
    do { \
        (ccd)->max_iterations = (unsigned long)-1; \
        (ccd)->epa_tolerance = 0.0001f; \
        (ccd)->mpr_tolerance = 0.0001f; \
        (ccd)->dist_tolerance = 1E-6f; \
    } while(0)

#define ZERO 0.0f
#define ONE 1.0f
#define COL_EPS FLT_EPSILON


struct GJKData{
	unsigned long max_iterations; //!< Maximal number of iterations
	float epa_tolerance;
	float mpr_tolerance; //!< Boundary tolerance for MPR algorithm
	float dist_tolerance;
};

typedef struct GJKData GJKData_t;

struct obj
{
	float x, y, z;
	float3 pos;
	float4 quat;
};
typedef struct obj obj_t;


__device__ void firstDir(const void *obj1, const void *obj2, float3 *dir);
__device__ void supportBox(const void *_obj, const float3 *_dir, float3 *v);
__device__ int GJKIntersect(const void *obj1, const void *obj2, const GJKData_t *data);
__device__ static int GJK(const void *obj1, const void *obj2,
				const GJKData_t *data, simplex_t *simplex);
__device__ void SupportCalc(const void *obj1, const void *obj2, const float3 *_dir, support_t *supp);
__device__ static int doSimplex(simplex_t *simplex, float3 *dir);
__device__ static int doSimplex2(simplex_t *simplex, float3 *dir);
__device__ static int doSimplex3(simplex_t *simplex, float3 *dir);
__device__ static int doSimplex4(simplex_t *simplex, float3 *dir);
__device__ float Vec3PointTriDist2(const float3 *P, const float3 *x0, const float3 *B, const float3 *C, float3 *witness);
__device__ float PointSegmentDist(const float3 *P, const float3 *x0, const float3 *b, float3 *witness);
__device__ void quatRotVec(float3 *v, const float4 *q);
__device__ int quatInvert2(float4 *dest, const float4 *src);
__device__ int quatInvert(float4 *q);
__device__ int Signum(float val);
__device__ void tripleCross(const float3 *a, const float3 *b,
				const float3 *c, float3 *d);
__device__ int IsZERO(float val);
__device__ int floatEq(float a, float b);
__device__ float absolute(float val);




void firstDir(const void *obj1, const void *obj2, float3 *dir) {
	dir->x = ONE;
	dir->y = ONE;
	dir->z = ZERO;
}

void supportBox(const void *_obj, const float3 *_dir, float3 *v)
{
	
		// assume that obj_t is user-defined structure that holds info about
		// object (in this case box: x, y, z, pos, quat - dimensions of box,
		// position and rotation)
		obj_t *obj = (obj_t *)_obj;
		float3 dir;
		float4 qinv;

		// apply rotation on direction vector
		float3Copy(&dir, _dir);
		quatInvert2(&qinv, &obj->quat);
		quatRotVec(&dir, &qinv);

		// compute support point in specified direction
		*v = make_float3(
			Signum(dir.x) * obj->x * 0.5f,
			Signum(dir.y) * obj->y * 0.5f,
			Signum(dir.z) * obj->z * 0.5f);
		// czlowiek to kubek q.e.d.
		// transform support point according to position and rotation of object
		quatRotVec(v, &obj->quat);
		float3Add(v,v, &obj->pos);

}

int GJKIntersect(const void *obj1, const void *obj2, const GJKData_t *data)
{
	simplex_t simplex;
	return GJK(obj1, obj2, data, &simplex) == 0;
}

static int GJK(const void *obj1, const void *obj2,
	const GJKData_t *data, simplex_t *simplex)
{
	unsigned long iterations;
	float3 dir; // direction vector
	support_t last; // last support point
	int do_simplex_res;

	// initialize simplex struct
	SimplexInit(simplex);

	// get first direction
	firstDir(obj1, obj2, &dir);
	// get first support point
	SupportCalc(obj1, obj2, &dir, &last);
	// and add this point to simplex as last one
	SimplexAdd(simplex, &last);

	// set up direction vector to as (O - last) which is exactly -last
	float3Copy(&dir, &last.v);
	float3Scale(&dir, -ONE);

	// start iterations
	for (iterations = 0UL; iterations < data->max_iterations; ++iterations) {
		// obtain support point
		SupportCalc(obj1, obj2, &dir, &last);

		// check if farthest point in Minkowski difference in direction dir
		// isn't somewhere before origin (the test on negative dot product)
		// - because if it is, objects are not intersecting at all.
		if (float3Dot(&last.v, &dir) < ZERO) {
			return -1; // intersection not found
		}

		// add last support vector to simplex
		SimplexAdd(simplex, &last);

		// if doSimplex returns 1 if objects intersect, -1 if objects don't
		// intersect and 0 if algorithm should continue
		do_simplex_res = doSimplex(simplex, &dir);
		if (do_simplex_res == 1) {
			return 0; // intersection found
		}
		else if (do_simplex_res == -1) {
			return -1; // intersection not found
		}

		if (IsZERO(float3Len(&dir))) {
			return -1; // intersection not found
		}
	}

	// intersection wasn't found
	return -1;
}

void SupportCalc(const void *obj1, const void *obj2, const float3 *_dir, support_t *supp)
{
	float3 dir;

	float3Copy(&dir, _dir);

	//wklejenie w v1 wyniku wywolania funkcji support1
	supportBox(obj1, &dir, &supp->v1);

	float3Scale(&dir, -ONE);
	//wklejenie w v2 wyniku wywolania funkcji support2
	supportBox(obj2, &dir, &supp->v2);

	//roznica minkowskiego
	float3Sub(&supp->v, &supp->v1, &supp->v2);
}


#pragma region doSimplexi

static int doSimplex(simplex_t *simplex, float3 *dir)
{
	if (SimplexSize(simplex) == 2) {
		// simplex contains segment only one segment
		return doSimplex2(simplex, dir);
	}
	else if (SimplexSize(simplex) == 3) {
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
	A = SimplexLast(simplex);
	// get the other point
	B = SimplexPoint(simplex, 0);
	// compute AB oriented segment
	float3Sub(&AB, &B->v, &A->v);
	// compute AO vector
	float3Copy(&AO, &A->v);
	float3Scale(&AO, -ONE);

	// dot product AB . AO
	dot = float3Dot(&AB, &AO);

	// check if origin doesn't lie on AB segment
	float3Cross(&tmp, &AB, &AO);
	if (IsZERO(float3Len(&tmp)) && dot > ZERO) {
		return 1;
	}

	// check if origin is in area where AB segment is
	if (IsZERO(dot) || dot < ZERO) {
		// origin is in outside are of A
		SimplexSet(simplex, 0, A);
		SimplexSetSize(simplex, 1);
		float3Copy(dir, &AO);
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
	const float3 origin = make_float3(0.f, 0.f, 0.f);
	const float3* originPtr = &origin;

	const support_t *A, *B, *C;
	float3 AO, AB, AC, ABC, tmp;
	float dot, dist;

	// get last added as A
	A = SimplexLast(simplex);
	// get the other points
	B = SimplexPoint(simplex, 1);
	C = SimplexPoint(simplex, 0);

	// check touching contact
	dist = Vec3PointTriDist2(originPtr, &A->v, &B->v, &C->v, NULL);
	if (IsZERO(dist)) {
		return 1;
	}

	// check if triangle is really triangle (has area > 0)
	// if not simplex can't be expanded and thus no itersection is found
	if (float3Eq(&A->v, &B->v) || float3Eq(&A->v, &C->v)) {
		return -1;
	}

	// compute AO vector
	float3Copy(&AO, &A->v);
	float3Scale(&AO, -ONE);

	// compute AB and AC segments and ABC vector (perpendircular to triangle)
	float3Sub(&AB, &B->v, &A->v);
	float3Sub(&AC, &C->v, &A->v);
	float3Cross(&ABC, &AB, &AC);

	float3Cross(&tmp, &ABC, &AC);
	dot = float3Dot(&tmp, &AO);
	if (IsZERO(dot) || dot > ZERO) {
		dot = float3Dot(&AC, &AO);
		if (IsZERO(dot) || dot > ZERO) {
			// C is already in place
			SimplexSet(simplex, 1, A);
			SimplexSetSize(simplex, 2);
			tripleCross(&AC, &AO, &AC, dir);
		}
		else {
		ccd_do_simplex3_45:
			dot = float3Dot(&AB, &AO);
			if (IsZERO(dot) || dot > ZERO) {
				SimplexSet(simplex, 0, B);
				SimplexSet(simplex, 1, A);
				SimplexSetSize(simplex, 2);
				tripleCross(&AB, &AO, &AB, dir);
			}
			else {
				SimplexSet(simplex, 0, A);
				SimplexSetSize(simplex, 1);
				float3Copy(dir, &AO);
			}
		}
	}
	else {
		float3Cross(&tmp, &AB, &ABC);
		dot = float3Dot(&tmp, &AO);
		if (IsZERO(dot) || dot > ZERO) {
			goto ccd_do_simplex3_45;
		}
		else {
			dot = float3Dot(&ABC, &AO);
			if (IsZERO(dot) || dot > ZERO) {
				float3Copy(dir, &ABC);
			}
			else {
				support_t Ctmp;
				SupportCopy(&Ctmp, C);
				SimplexSet(simplex, 0, B);
				SimplexSet(simplex, 1, &Ctmp);

				float3Copy(dir, &ABC);
				float3Scale(dir, -ONE);
			}
		}
	}

	return 0;
}

static int doSimplex4(simplex_t *simplex, float3 *dir)
{
	const float3 origin = make_float3(0.f, 0.f, 0.f);
	const float3* originPtr = &origin;

	const support_t *A, *B, *C, *D;
	float3 AO, AB, AC, AD, ABC, ACD, ADB;
	int B_on_ACD, C_on_ADB, D_on_ABC;
	int AB_O, AC_O, AD_O;
	float dist;

	// get last added as A
	A = SimplexLast(simplex);
	// get the other points
	B = SimplexPoint(simplex, 2);
	C = SimplexPoint(simplex, 1);
	D = SimplexPoint(simplex, 0);

	// check if tetrahedron is really tetrahedron (has volume > 0)
	// if it is not simplex can't be expanded and thus no intersection is
	// found
	dist = Vec3PointTriDist2(&A->v, &B->v, &C->v, &D->v, NULL);
	if (IsZERO(dist)) {
		return -1;
	}

	// check if origin lies on some of tetrahedron's face - if so objects
	// intersect
	dist = Vec3PointTriDist2(originPtr, &A->v, &B->v, &C->v, NULL);
	if (IsZERO(dist))
		return 1;
	dist = Vec3PointTriDist2(originPtr, &A->v, &C->v, &D->v, NULL);
	if (IsZERO(dist))
		return 1;
	dist = Vec3PointTriDist2(originPtr, &A->v, &B->v, &D->v, NULL);
	if (IsZERO(dist))
		return 1;
	dist = Vec3PointTriDist2(originPtr, &B->v, &C->v, &D->v, NULL);
	if (IsZERO(dist))
		return 1;

	// compute AO, AB, AC, AD segments and ABC, ACD, ADB normal vectors
	float3Copy(&AO, &A->v);
	float3Scale(&AO, -ONE);
	float3Sub(&AB, &B->v, &A->v);
	float3Sub(&AC, &C->v, &A->v);
	float3Sub(&AD, &D->v, &A->v);
	float3Cross(&ABC, &AB, &AC);
	float3Cross(&ACD, &AC, &AD);
	float3Cross(&ADB, &AD, &AB);

	// side (positive or negative) of B, C, D relative to planes ACD, ADB
	// and ABC respectively
	B_on_ACD = Signum(float3Dot(&ACD, &AB));
	C_on_ADB = Signum(float3Dot(&ADB, &AC));
	D_on_ABC = Signum(float3Dot(&ABC, &AD));

	// whether origin is on same side of ACD, ADB, ABC as B, C, D
	// respectively
	AB_O = Signum(float3Dot(&ACD, &AO)) == B_on_ACD;
	AC_O = Signum(float3Dot(&ADB, &AO)) == C_on_ADB;
	AD_O = Signum(float3Dot(&ABC, &AO)) == D_on_ABC;

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
		SimplexSet(simplex, 2, A);
		SimplexSetSize(simplex, 3);
	}
	else if (!AC_O) {
		// C is farthest
		SimplexSet(simplex, 1, D);
		SimplexSet(simplex, 0, B);
		SimplexSet(simplex, 2, A);
		SimplexSetSize(simplex, 3);
	}
	else { // (!AD_O)
		SimplexSet(simplex, 0, C);
		SimplexSet(simplex, 1, B);
		SimplexSet(simplex, 2, A);
		SimplexSetSize(simplex, 3);
	}

	return doSimplex3(simplex, dir);
}



#pragma endregion


float Vec3PointTriDist2(const float3 *P,const float3 *x0, const float3 *B,const float3 *C,float3 *witness)

{
	// Computation comes from analytic expression for triangle (x0, B, C)
	//      T(s, t) = x0 + s.d1 + t.d2, where d1 = B - x0 and d2 = C - x0 and
	// Then equation for distance is:
	//      D(s, t) = | T(s, t) - P |^2
	// This leads to minimization of quadratic function of two variables.
	// The solution from is taken only if s is between 0 and 1, t is
	// between 0 and 1 and t + s < 1, otherwise distance from segment is
	// computed.

	float3 d1, d2, a;
	float u, v, w, p, q, r, d;
	float s, t, dist, dist2;
	float3 witness2;

	float3Sub(&d1, B, x0);
	float3Sub(&d2, C, x0);
	float3Sub(&a, x0, P);

	u = float3Dot(&a, &a);
	v = float3Dot(&d1, &d1);
	w = float3Dot(&d2, &d2);
	p = float3Dot(&a, &d1);
	q = float3Dot(&a, &d2);
	r = float3Dot(&d1, &d2);

	d = w * v - r * r;
	if (IsZERO(d)) {
		// To avoid division by zero for zero (or near zero) area triangles
		s = t = -1.f;
	}
	else {
		s = (q * r - w * p) / d;
		t = (-s * r - q) / w;
	}

	if ((IsZERO(s) || s > ZERO)
		&& (floatEq(s, ONE) || s < ONE)
		&& (IsZERO(t) || t > ZERO)
		&& (floatEq(t, ONE) || t < ONE)
		&& (floatEq(t + s, ONE) || t + s < ONE)) {

		if (witness) {
			float3Scale(&d1, s);
			float3Scale(&d2, t);
			float3Copy(witness, x0);
			float3Add(witness, witness, &d1);
			float3Add(witness, witness, &d2);

			dist = float3Dist(witness, P);
		}
		else {
			dist = s * s * v;
			dist += t * t * w;
			dist += 2.f  * s * t * r;
			dist += 2.f * s * p;
			dist += 2.f  * t * q;
			dist += u;
		}
	}
	else {
		dist = PointSegmentDist(P, x0, B, witness);

		dist2 = PointSegmentDist(P, x0, C, &witness2);
		if (dist2 < dist) {
			dist = dist2;
			if (witness)
				float3Copy(witness, &witness2);
		}

		dist2 = PointSegmentDist(P, B, C, &witness2);
		if (dist2 < dist) {
			dist = dist2;
			if (witness)
				float3Copy(witness, &witness2);
		}
	}

	return dist;
}

float PointSegmentDist(const float3 *P, const float3 *x0, const float3 *b, float3 *witness)
{
	// The computation comes from solving equation of segment:
	//      S(t) = x0 + t.d
	//          where - x0 is initial point of segment
	//                - d is direction of segment from x0 (|d| > 0)
	//                - t belongs to <0, 1> interval
	// 
	// Than, distance from a segment to some point P can be expressed:
	//      D(t) = |x0 + t.d - P|^2
	//          which is distance from any point on segment. Minimization
	//          of this function brings distance from P to segment.
	// Minimization of D(t) leads to simple quadratic equation that's
	// solving is straightforward.
	//
	// Bonus of this method is witness point for free.

	float dist, t;
	float3 d, a;

	// direction of segment
	float3Sub(&d, b, x0);

	// precompute vector from P to x0
	float3Sub(&a, x0, P);

	t = -1.f * float3Dot(&a, &d);
	t /= float3Len(&d);

	if (t < ZERO || IsZERO(t)) {
		dist = float3Dist(x0, P);
		if (witness)
			float3Copy(witness, x0);
	}
	else if (t > ONE || floatEq(t, ONE)) {
		dist = float3Dist(b, P);
		if (witness)
			float3Copy(witness, b);
	}
	else {
		if (witness) {
			float3Copy(witness, &d);
			float3Scale(witness, t);
			float3Add(witness,witness, x0);
			dist = float3Dist(witness, P);
		}
		else {
			// recycling variables
			float3Scale(&d, t);
			float3Add(&d, &d, &a);
			dist = float3Len(&d);
		}
	}

	return dist;
}

void quatRotVec(float3 *v, const float4 *q)
{
	// original version: 31 mul + 21 add
	// optimized version: 18 mul + 12 add
	// formula: v = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
	float cross1_x, cross1_y, cross1_z, cross2_x, cross2_y, cross2_z;
	float x, y, z, w;
	float vx, vy, vz;

	vx = v->x;
	vy = v->y;
	vz = v->z;

	w = q->w;
	x = q->x;
	y = q->y;
	z = q->z;

	cross1_x = y * vz - z * vy + w * vx;
	cross1_y = z * vx - x * vz + w * vy;
	cross1_z = x * vy - y * vx + w * vz;
	cross2_x = y * cross1_z - z * cross1_y;
	cross2_y = z * cross1_x - x * cross1_z;
	cross2_z = x * cross1_y - y * cross1_x;
	*v = make_float3( vx + 2 * cross2_x, vy + 2 * cross2_y, vz + 2 * cross2_z);
}

int quatInvert2(float4 *dest, const float4 *src)
{
	float4Copy(dest, src);
	return quatInvert(dest);
}

int quatInvert(float4 *q)
{
	float len2 = magnitudeSqr(*q);
	if (len2 < FLT_EPSILON)
		return -1;

	len2 = ONE / len2;

	q->x = -q->x * len2;
	q->y = -q->y * len2;
	q->z = -q->z * len2;
	q->w = q->w * len2;

	return 0;
}

#pragma region  inlines

int Signum(float val)
{
	if (IsZERO(val)) {
		return 0;
	}
	else if (val < ZERO) {
		return -1;
	}
	return 1;
}


void tripleCross(const float3 *a, const float3 *b,
	const float3 *c, float3 *d)
{
	float3 e;
	float3Cross(&e, a, b);
	float3Cross(d, &e, c);
}


int IsZERO(float val)
{
	return absolute(val) < COL_EPS;
}

int floatEq(float a, float b)
{
	return a == b;
}

float absolute(float val)
{
	return val > 0 ? val : -val;
}

#pragma endregion

