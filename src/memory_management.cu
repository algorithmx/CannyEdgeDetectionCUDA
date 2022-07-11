/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *  GNU GENERAL PUBLIC LICENSE
 *  Version 3, 29 June 2007
 *
 *  Author: Yunlong Lian @algorithmx 
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include "println.hpp"

#define gpuErrchk(ans) { GPU_error_check((ans), __FILE__, __LINE__); }

inline void GPU_error_check(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess)
   {
	  std::cerr << "GPU_error_check: code=" << \
					cudaGetErrorString(code) << ", in file " << \
					file << " line " << line << std::endl;
	  if (abort)
	  	exit(code);
   }
}

// ------------- free -------------

template <typename T>
__host__ void _cu_mem_free_char(T * cu_device_arr) {
	if (cu_device_arr != NULL)
		gpuErrchk( cudaFree((void*)cu_device_arr) );
}


template <typename T>
__host__ void _cu_async_mem_free_char(T * cu_device_arr, cudaStream_t &stream) {
	if (cu_device_arr!=NULL)
		gpuErrchk( cudaFreeAsync((void*)cu_device_arr, stream) ) ;
}


template <typename T>
__host__ void _mem_free_char_pinned(T * arr) {
	if (arr!=NULL)
		gpuErrchk( cudaFreeHost((void*)arr) );
}


template <typename T>
__host__ void _mem_free_char(T * arr) {
	try {
		if (arr != NULL)
			delete[] arr; 
	}
	catch(const std::exception& e)  {
		errorln(e.what()); 
		errorln("Memory error in _mem_free_char().");
		exit(1) ;
	}
}


// ------------- allocate -------------

template <typename T>
__host__ T * _cu_mem_alloc_char(int size) {
	T * d_arr = NULL ;
	gpuErrchk( cudaMalloc((void **)&d_arr, size*sizeof(T)) ) ;
	return d_arr ;
}


template <typename T>
__host__ T * _cu_async_mem_alloc_char(int size, cudaStream_t &stream) {
	T * d_arr = NULL ;
	gpuErrchk( cudaMallocAsync((void **)&d_arr, size*sizeof(T), stream) );
	return d_arr ;
};


template <typename T>
T * _mem_alloc_char(int size) {
	T *LOCAL_ARRAY = NULL ;
	if (size > 0) {
		try {
			LOCAL_ARRAY = new T[ size ];
			for (int p=0; p<size; ++p)
				LOCAL_ARRAY[p] = T() ;
		}
		catch(const std::exception& e) {
			errorln(e.what()); 
			errorln("Memory error in _mem_alloc_char(). Exit.");
			exit(1) ;
		}
	}
	return LOCAL_ARRAY ;
}


template <typename T>
T * _mem_alloc_char_pinned(int size) {
	T *LOCAL_ARRAY = NULL ;
	if (size > 0) {
		gpuErrchk( cudaMallocHost((void**)&LOCAL_ARRAY, size*sizeof(T)) );
		for (int p=0; p<size; ++p)
			LOCAL_ARRAY[p] = T() ;
	}
	return LOCAL_ARRAY ;
}


// ------------- cpoy -------------

template <typename T>
__host__ void _cu_mem_copy(
	T *src, 
	T *dst, 
	int size, 
	cudaMemcpyKind cudaMemcpy_To_
) {
	gpuErrchk( cudaMemcpy((void*)dst, (void*)src, size*sizeof(T), cudaMemcpy_To_) ) ;
}


template <typename T>
__host__ void _cu_async_mem_copy(
	T *src, 
	T *dst, 
	int size, 
	cudaMemcpyKind cudaMemcpy_To_, 
	cudaStream_t &stream
) {
	gpuErrchk( cudaMemcpyAsync((void*)dst, (void*)src, size*sizeof(T), cudaMemcpy_To_, stream) ) ;
}


template <typename T>
void _mem_copy(T *src, T *dst, int size)  {
	if (size > 0) {
		try {
			std::memcpy((void*)dst, (void*)src, size*sizeof(T));
		}
		catch(const std::exception& e) {
			errorln(e.what()); 
			errorln("Memory error in _mem_copy(). Exit.");
			exit(1);
		}
	}
}
