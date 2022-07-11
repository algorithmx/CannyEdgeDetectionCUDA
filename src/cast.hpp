#pragma once

// don't say that word ...
// I know this has to be simplified

#define max2(a, b)  ((a>b) ? (a) : (b))

#define min2(a, b)  ((a<b) ? (a) : (b))

inline unsigned char char_uchar(char c) {
	return static_cast<unsigned char>(c);
}

__device__ inline unsigned char cu_char_uchar(char c)  {
	return static_cast<unsigned char>(c);
}

inline char uchar_char(unsigned char c) {
	return static_cast<char>(c);
}

__device__ inline char cu_uchar_char(unsigned char c) {
	return static_cast<char>(c);
}

inline int char_uchar_int(char c) {
	return static_cast<int>(char_uchar(c));
}

__device__ inline int  cu_char_uchar_int(char c) {
	return static_cast<int>(cu_char_uchar(c));
}

inline unsigned int char_uchar_uint(char c) {
	return static_cast<unsigned int>(char_uchar(c));
}

inline char uint_uchar_char(unsigned int i) {
	return uchar_char(static_cast<unsigned char>(min2(255u,i)));
}

__device__ inline char cu_uint_uchar_char(unsigned int i) {
	return cu_uchar_char(static_cast<unsigned char>(min2(255u,i)));
}

inline unsigned char int_uchar(int i) {
	return static_cast<unsigned char>(min2(255,max2(i,0)));
}

__device__ inline unsigned char cu_int_uchar(int i) {
	return static_cast<unsigned char>(min2(255,max2(i,0))); 
}

inline char int_char(int i) {
	return static_cast<char>(min2(127,max2(i,-128))); 
}

inline char int_uchar_char(int i) {
	return static_cast<char>(int_uchar(i)); 
}

__device__ inline char cu_int_uchar_char(int i) {
	return static_cast<char>(cu_int_uchar(i)); 
}