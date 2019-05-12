#pragma once
#include "support.h"

struct simplex {
	support_t ps[4];
	int last; //!< index of last added point
};
typedef struct simplex simplex_t;

void SimplexInit(simplex_t *s);
int SimplexSize(const simplex_t *s);
const support_t *SimplexLast(const simplex_t *s);
const support_t *SimplexPoint(const simplex_t *s, int idx);
support_t *SimplexPointW(simplex_t *s, int idx);

void SimplexAdd(simplex_t *s, const support_t *v);
void SimplexSet(simplex_t *s, size_t pos, const support_t *a);
void SimplexSetSize(simplex_t *s, int size);
void SimplexSwap(simplex_t *s, size_t pos1, size_t pos2);

void SimplexInit(simplex_t *s)
{
	s->last = -1;
}

int SimplexSize(const simplex_t *s)
{
	return s->last + 1;
}

const support_t *SimplexLast(const simplex_t *s)
{
	return SimplexPoint(s, s->last);
}

const support_t *SimplexPoint(const simplex_t *s, int idx)
{
	// here is no check on boundaries
	return &s->ps[idx];
}
support_t *SimplexPointW(simplex_t *s, int idx)
{
	return &s->ps[idx];
}

void SimplexAdd(simplex_t *s, const support_t *v)
{
	// here is no check on boundaries in sake of speed
	++s->last;
	SupportCopy(s->ps + s->last, v);
}

void SimplexSet(simplex_t *s, size_t pos, const support_t *a)
{
	SupportCopy(s->ps + pos, a);
}

void SimplexSetSize(simplex_t *s, int size)
{
	s->last = size - 1;
}

void SimplexSwap(simplex_t *s, size_t pos1, size_t pos2)
{
	support_t supp;

	SupportCopy(&supp, &s->ps[pos1]);
	SupportCopy(&s->ps[pos1], &s->ps[pos2]);
	SupportCopy(&s->ps[pos2], &supp);
}
