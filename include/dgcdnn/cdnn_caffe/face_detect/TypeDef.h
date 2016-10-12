
#ifndef H_TYPEDEF_H
#define H_TYPEDEF_H
#pragma once

#include <stdio.h>
#include <stdlib.h>

#ifdef WIN32
 
typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned __int64 uint64;
typedef unsigned __int32 uint32;
typedef __int32 int32;
typedef __int64 int64;

#else

#include <stdint.h>
typedef unsigned char uchar;
typedef unsigned int uint;
typedef uint64_t uint64;
typedef uint32_t uint32;
typedef int64_t int64;
typedef int32_t int32;

#endif

//#define NULL 0

typedef float LUT_BIN_TYPE;

#if !defined(MIN)
#define MIN(x,y) ((x)<=(y)?(x):(y))
#endif
#if !defined(MAX)
#define MAX(x,y) ((x)>=(y)?(x):(y))
#endif
#define ABS(x) ((x)>=0?(x):(-x))
//#define BOUND(val, lb, ub) ((val)<(lb)?(lb):((val)>(ub)?(ub):(val)))
#define LEFT_SHIFT(x,n) (((n)>=0)?((x)<<(n)):((x)>>-(n)))

const int MAX_THREAD_NUM = 16;
#define PI 3.1415926

#ifndef IN
#define IN
#endif
#ifndef OUT
#define OUT
#endif
#ifndef BUF
#define BUF
#endif


template <class T>
struct TSRect
{
	T l, t, r, b;	//left, top, right and bottom
	T CalAre() {return (r-l)*(b-t);}
	TSRect() {};
	TSRect(T l, T t, T r, T b) {this->l=l; this->t=t; this->r=r; this->b=b;}

	TSRect<T>& operator = (const TSRect<T> &rect) {	this->l = rect.l;		this->t = rect.t; 		this->r = rect.r; 		this->b = rect.b; return *this;}

};

template <class T>
 bool GetIntersectionRect(const TSRect<T> &r1, const TSRect<T> &r2, TSRect<T> &rect)
{
	T l = MAX(r1.l, r2.l);
	T r = MIN(r1.r, r2.r);
	if(l>=r) return false;

	T t = MAX(r1.t, r2.t);
	T b = MIN(r1.b, r2.b);
	if(t>=b) return false;

	rect.l = l;
	rect.r = r;
	rect.t = t;
	rect.b = b;
	return true;
}

template <class T>
float CalOverlapRatio(TSRect<T> &r1, TSRect<T> &r2,float& r1_ratio, float& r2_ratio)
{
	
	TSRect<T> r;
	if(!GetIntersectionRect(r1, r2, r)){
		r1_ratio=r2_ratio=0.0;
		return 0.0;
	}else{
		r1_ratio = r.CalAre()/r1.CalAre();
		r2_ratio = r.CalAre()/r2.CalAre();
		return float(r.CalAre())/(r1.CalAre()+r2.CalAre()-r.CalAre());
	}
}

template <class T>
bool IsR1WithinR2Rect(TSRect<T> &r1, TSRect<T> &r2)
{
	if(r1.l>=r2.l && r1.r<=r2.r && r1.t>=r2.t && r1.b<=r2.b)
		return true;
	else
		return false;
}

template <class T1, class T2, class T3>
struct TTriple
{
	T1 first;
	T2 second;
	T3 third;
	TTriple() {}
	TTriple(T1 f, T2 s, T3 t) {first = f; second = s; third = t;}
}; 
 

#define REPORT_ERROR_POSITION {fprintf(stderr, "Error happens at line %d of %s\n", __LINE__, __FILE__); exit(-1);}

#endif

