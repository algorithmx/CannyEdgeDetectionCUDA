/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *  GNU GENERAL PUBLIC LICENSE
 *  Version 3, 29 June 2007
 *
 *  Author: Yunlong Lian @algorithmx 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#include <iostream>
#include <atomic>
#include <cuda_runtime.h>
#include "println.hpp"
#include "timing.hpp"
#include "memory_management.cu"


struct ExecutionPolicy  {
	dim3 blockSize;
	dim3 gridSize;
	ExecutionPolicy(): blockSize(dim3(1,1,1)), gridSize(dim3(1,1,1)) { } 
	ExecutionPolicy(int blk, int grd): blockSize(dim3(blk,1,1)), gridSize(dim3(grd,1,1)) { } 
	ExecutionPolicy(dim3 blk, dim3 grd): blockSize(blk), gridSize(grd) { } 
	~ExecutionPolicy() { } 
};


// class dependency :
//
// Mem<T>
// | ... CPU<T>
//   ... data : _MEM, _SIZE
// 
//         | ... GPU<T>
//           ... additional data: _MEM_GPU
//
//
// | ... CPUpinned<T>
//   ... data : _MEM, _SIZE
//    
//         | ... GPUpinned<T>
//           ... additional data: _MEM_GPU
//     
//         | ... GPUstream<T>
//           ... additional data: _MEM_GPU
//          


template <class T = char>
class Mem
{
	public: 

		Mem() = delete ; // prevent undefined behavior

		Mem(int nbytes):  _SIZE(nbytes),__in_operation(false) {
			/* do nothing */
		}

		Mem(const Mem<T> & M) : _SIZE(M._SIZE), __in_operation(false) {
			/* do nothing */
		}

		Mem<T> &operator=(Mem<T> const& M);

		virtual ~Mem() { /* do nothing */ }

		T get_byte(int pos) const {
			return (pos>=0 && pos<_SIZE) ? get_byte_unsafe(pos) : T() ; 
		}

		T get_byte_unsafe(int pos) const { return _MEM[pos] ; }

		void set_byte(int pos, T c) {
			if (pos>=0 && pos<_SIZE)
				set_byte_unsafe(pos, c); 
		}

		void set_byte_unsafe(int pos, T c) { _MEM[pos] = c; } 

		virtual void CopyIn(T* src) = 0 ;

		virtual void CopyOut(T* dst) = 0 ;

		virtual void cuCopyIn(T* src)  = 0 ;

		virtual void cuCopyOut(T* src) = 0 ;

		virtual void sync(bool up) = 0 ;

		template <typename... Arguments>
		void execute_kernel(
				T* dst, 
				const ExecutionPolicy& ExPol, 
				void (*f)(Arguments...), 
				Arguments... args ) {}


	protected:

		T *_MEM = NULL ;

		int _SIZE = 0;

		std::atomic<bool> __in_operation;  // operation lock

		void _lock(void) {
			while(__in_operation ) 
				wait();
			__in_operation = true; 
		}

		void _unlock(void) { __in_operation = false ; }

		// memory management, to be specified by the derived classes
		// according to memory types: CPU pageable, CPU pinned or GPU
		virtual void _mem_alloc() = 0 ;
		virtual void _mem_free() = 0 ;

};


//////////////////////////////////////////////////////////////

// odinary memory allocator

template<class T = char>
class CPU : public Mem<T> {
	public:

		CPU(int nbytes) : Mem<T>(nbytes) { _mem_alloc(); }

		CPU(const CPU<T> & M) : Mem<T>(M)  { 
			_mem_free();
			_mem_alloc();
			CopyIn(M._MEM);
		}

		CPU<T> &operator=(CPU<T> const& M)  {
			if (this == &M) { return *this; }
			//
			_mem_free();
			Mem<T>::_SIZE = M._SIZE;
			_mem_alloc();
			CopyIn(M._MEM);
			//
			return *this;
		}

		~CPU(){ _mem_free(); }

		void CopyIn(T* src) { 
			Mem<T>::_lock(); 
			_mem_copy<T>(src, Mem<T>::_MEM, Mem<T>::_SIZE);
			Mem<T>::_unlock(); 
		}

		void CopyOut(T* dst) {
			Mem<T>::_lock(); 
			_mem_copy<T>(Mem<T>::_MEM, dst, Mem<T>::_SIZE); // no need to sync()
			Mem<T>::_unlock(); 
		}

		void cuCopyIn(T* src) {}

		void cuCopyOut(T* src) {}

		void sync(bool up = false) {}

		// GOAL : expose  _MEM  and  _SIZE  to external cpu_function(...)  
		template <typename... Arguments>
		void execute_kernel(
			T* dst, 
			const ExecutionPolicy& dummy_ExPol, 
			void (*cpu_function)(T*, T*, int, Arguments...), 
			Arguments... args
		) {
			// convention for cpu_function(dst, src, mem_index_size, other arguments...) argument list
			cpu_function(dst, Mem<T>::_MEM, Mem<T>::_SIZE, args...);
		}

		template <typename... Arguments>
		void execute_kernel(
			T* dst, 
			void (*cpu_function)(T*, T*, int, Arguments...), 
			Arguments... args
		) {
			execute_kernel(dst, ExecutionPolicy(), cpu_function, args...);
		}

		template <typename... Arguments>
		void execute_kernel(
			void (*cpu_function)(T*, T*, int, Arguments...), 
			Arguments... args
		) {
			T* DST = _mem_alloc_char<T>(Mem<T>::_SIZE) ;
			execute_kernel(DST, ExecutionPolicy(), cpu_function, args...);
			Mem<T>::CopyIn(DST);
			_mem_free_char<T>(DST);
		}


	protected:

		void _mem_alloc() {
			Mem<T>::_MEM = _mem_alloc_char<T>(Mem<T>::_SIZE);
		}

		void _mem_free() {
			Mem<T>::_lock(); 
			_mem_free_char<T> (Mem<T>::_MEM); 
			Mem<T>::_unlock(); 
		}
};


//////////////////////////////////////////////////////////////

// special memory allocator


template <class T = char>
class CPUpinned : public Mem<T> {
	public:

		CPUpinned(int nbytes) : Mem<T>(nbytes) { _mem_alloc(); }

		CPUpinned(const CPUpinned<T> & M) : Mem<T>(M) {
			//println("CPUpinned copy constructor called.");
			_mem_free();
			_mem_alloc();
			CopyIn(M._MEM);
		}

		CPUpinned<T> &operator=(CPUpinned<T> const& M) {
			if (this == &M) { return *this; }
			//
			_mem_free();
			Mem<T>::_SIZE = M._SIZE;
			_mem_alloc();
			CopyIn(M._MEM);
			//
			return *this;
		}

		~CPUpinned() { _mem_free(); }

		void CopyIn(T* src) { 
			_cu_mem_copy<T> (src, Mem<T>::_MEM, Mem<T>::_SIZE, cudaMemcpyHostToHost); 
		}

		void CopyOut(T* dst) { 
			Mem<T>::_lock();
			_cu_mem_copy<T> (Mem<T>::_MEM, dst, Mem<T>::_SIZE, cudaMemcpyHostToHost); 
			Mem<T>::_unlock();
		}

		void cuCopyIn(T* src) {}

		void cuCopyOut(T* src) {}

		void sync(bool up = false) {}


	protected:

		void _mem_alloc() {
			Mem<T>::_MEM = _mem_alloc_char_pinned<T> (Mem<T>::_SIZE);
		}

		void _mem_free() { 
			Mem<T>::_lock(); 
			_mem_free_char_pinned<T> (Mem<T>::_MEM); 
			Mem<T>::_unlock(); 
		}
};


//////////////////////////////////////////////////////////////

// use default stream, non-pinned memory

template <class T = char>
class GPU : public CPU<T> {

	public:

		GPU (int size) : CPU<T>(size) { _gpu_mem_alloc(); }

		GPU (const GPU<T> & M) :CPU<T>(M) {
			_gpu_mem_free();
			_gpu_mem_alloc();
			sync(true) ; // upload
		}

		GPU<T> &operator=(GPU<T> const& M) {
			if (this == &M) { return *this; }
			//
			CPU<T>::_mem_free();
			_gpu_mem_free();
			CPU<T>::_SIZE = M._SIZE;
			CPU<T>::_mem_alloc();
			CopyIn(M._MEM);
			_gpu_mem_alloc();
			cuCopyIn(M._MEM_GPU);
			sync(true) ; // upload
			//
			return *this;
		}

		~GPU() { _gpu_mem_free(); }

		void CopyIn(T* src) {
			CPU<T>::_lock(); 
			_mem_copy<T>(src, CPU<T>::_MEM, CPU<T>::_SIZE);
			CPU<T>::_unlock();
			sync(true) ;
		}

		void CopyOut(T* dst) {
			CPU<T>::_lock(); 
			_mem_copy<T>(CPU<T>::_MEM, dst, CPU<T>::_SIZE);
			CPU<T>::_unlock(); 
		}

		void cuCopyIn(T* src) {
			CPU<T>::_lock(); 
			_cu_mem_copy<T> (src, _MEM_GPU, CPU<T>::_SIZE, cudaMemcpyDeviceToDevice); 
			CPU<T>::_unlock();
		}

		void cuCopyOut(T* dst) {
			CPU<T>::_lock();
			_cu_mem_copy<T> (_MEM_GPU, dst, CPU<T>::_SIZE, cudaMemcpyDeviceToDevice); 
			CPU<T>::_unlock();
		}

		void sync(bool up = false) {
			if (up) { _upload(); }
			else { _download(); }
		}

		// GOAL : expose  _MEM_GPU  and  _SIZE  to external cuda_kernel(...)  
		template <typename... Arguments>
		void execute_kernel(
			T* DST, 
			const ExecutionPolicy& ExPol, 
			void (*cuda_kernel)(T*, T*, int, Arguments...), 
			Arguments... args
		) {
			// references:
			// https://eli.thegreenplace.net/2014/variadic-templates-in-c/
			// https://en.cppreference.com/w/cpp/language/parameter_pack
			// https://developer.nvidia.com/blog/cplusplus-11-in-cuda-variadic-templates/
			// https://stackoverflow.com/questions/65886397/c-function-with-variable-number-and-types-of-arguments-as-argument-of-another
			//
			// convention for cuda_kernel(DST, src, mem_index_size, other arguments...) argument list

			cuda_kernel<<<ExPol.gridSize, ExPol.blockSize>>>(DST, _MEM_GPU, CPU<T>::_SIZE, args...);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				println("    error in GPU<T>.cuda_kernel() ... ");
				std::cerr << cudaGetErrorString(err) << std::endl;
				exit(1);
			}
		}

		// special version:
		// only if cuda_kernel intends to write out to TMP 
		template <typename... Arguments>
		void execute_kernel(
			const ExecutionPolicy& ExPol, 
			void (*cuda_kernel)(T*, T*, int, Arguments...), 
			Arguments... args 
		) {
			T* TMP = _cu_mem_alloc_char<T>(CPU<T>::_SIZE) ;
			execute_kernel(TMP, ExecutionPolicy(), cuda_kernel, args...);
			CPU<T>::cuCopyIn(TMP);
			CPU<T>::sync(false); // download
			_cu_mem_free_char<T>(TMP);
		}


	protected:

		T * _MEM_GPU = NULL;

		void _download() {
			CPU<T>::_lock(); 
			_cu_mem_copy<T>(_MEM_GPU, CPU<T>::_MEM, CPU<T>::_SIZE, cudaMemcpyDeviceToHost);
			CPU<T>::_unlock();
		}

		void _upload() {
			CPU<T>::_lock(); 
			_cu_mem_copy<T>(CPU<T>::_MEM, _MEM_GPU, CPU<T>::_SIZE, cudaMemcpyHostToDevice); 
			CPU<T>::_unlock();
		}

		void _gpu_mem_alloc() {
			_MEM_GPU = _cu_mem_alloc_char<T>(CPU<T>::_SIZE);
		}

		void _gpu_mem_free() {
			CPU<T>::_lock(); 
			_cu_mem_free_char<T>(_MEM_GPU); 
			CPU<T>::_unlock(); 
		}
};


//////////////////////////////////////////////////////////////

// use independent stream

template <class T = char>
class GPUpinned : public CPUpinned<T>  {

	public:

		GPUpinned (int size) : CPUpinned<T>(size) { _gpu_mem_alloc(); }; 

		GPUpinned (const GPUpinned<T> & M) :CPUpinned<T>(M) { 
			//println("GPUpinned copy constructor called.");
			_gpu_mem_free();
			_gpu_mem_alloc();
			sync(true) ; // upload
		}

		GPUpinned<T> &operator=(GPUpinned<T> const& M) {
			if (this == &M) { return *this; }
			//
			CPUpinned<T>::_mem_free();
			_gpu_mem_free();
			CPUpinned<T>::_SIZE = M._SIZE;
			CPUpinned<T>::_mem_alloc();
			CPUpinned<T>::CopyIn(M._MEM);
			_gpu_mem_alloc();
			CPUpinned<T>::cuCopyIn(M._MEM_GPU);
			sync(true) ; // upload
			//
			return *this;
		}

		~GPUpinned() { 
			_gpu_mem_free();
		}

		void CopyIn(T* src) {
			CPUpinned<T>::_lock(); 
			_cu_mem_copy<T> (src, CPUpinned<T>::_MEM, CPUpinned<T>::_SIZE, cudaMemcpyHostToHost); 
			CPUpinned<T>::_unlock();
			sync(true) ; // upload
		}

		void cuCopyIn(T* src) {
			CPUpinned<T>::_lock(); 
			_cu_mem_copy<T> (src, _MEM_GPU, CPUpinned<T>::_SIZE, cudaMemcpyDeviceToDevice); 
			CPUpinned<T>::_unlock();
		}

		void cuCopyOut(T* dst) {
			CPUpinned<T>::_lock();
			_cu_mem_copy<T> (_MEM_GPU, dst, CPUpinned<T>::_SIZE, cudaMemcpyDeviceToDevice); 
			CPUpinned<T>::_unlock();
		}

		void CopyOut(T* dst) {
			CPUpinned<T>::_lock();
			_cu_mem_copy<T> (CPUpinned<T>::_MEM, dst, CPUpinned<T>::_SIZE, cudaMemcpyHostToHost);
			CPUpinned<T>::_unlock();
		}

		void sync(bool up = false) {
			if (up) { _upload(); }
			else { _download(); }
		}


	protected:

		T * _MEM_GPU = NULL ;

		void _download() { 
			CPUpinned<T>::_lock(); 
			_cu_mem_copy<T>(_MEM_GPU, CPUpinned<T>::_MEM, CPUpinned<T>::_SIZE, cudaMemcpyDeviceToHost); 
			CPUpinned<T>::_unlock();
		}

		void _upload() {
			CPUpinned<T>::_lock();  
			_cu_mem_copy<T>(CPUpinned<T>::_MEM, _MEM_GPU, CPUpinned<T>::_SIZE, cudaMemcpyHostToDevice);
			CPUpinned<T>::_unlock();
		}

		void _gpu_mem_alloc() { 
			_MEM_GPU = _cu_mem_alloc_char<T>(CPUpinned<T>::_SIZE);
		}

		void _gpu_mem_free() { 
			CPUpinned<T>::_lock();  
			_cu_mem_free_char<T>(_MEM_GPU);
			CPUpinned<T>::_unlock();
		}

};


// use independent stream

template <class T = char>
class GPUstream : public GPUpinned<T>  {

	public:

		GPUstream (int size) : GPUpinned<T>(size) { 
			gpuErrchk(cudaStreamCreateWithFlags(&_CUDA_STREAM, cudaStreamNonBlocking));
		}

		GPUstream (const GPUstream<T> & M) : GPUpinned<T>(M) {
			if (_CUDA_STREAM != NULL)
				gpuErrchk(cudaStreamDestroy(_CUDA_STREAM));
			gpuErrchk(cudaStreamCreateWithFlags(&_CUDA_STREAM, cudaStreamNonBlocking));
		}

		GPUstream<T> &operator=(GPUstream<T> const& M) {
			if (this == &M) { return *this; }
			//
			if (_CUDA_STREAM != NULL)
				gpuErrchk(cudaStreamDestroy(_CUDA_STREAM));
			gpuErrchk(cudaStreamCreateWithFlags(&_CUDA_STREAM, cudaStreamNonBlocking));
			GPUpinned<T>::_mem_free();
			GPUpinned<T>::_gpu_mem_free();
			GPUpinned<T>::_SIZE = M._SIZE;
			GPUpinned<T>::_mem_alloc();
			GPUpinned<T>::CopyIn(M._MEM);
			GPUpinned<T>::_gpu_mem_alloc();
			GPUpinned<T>::cuCopyIn(M._MEM_GPU);
			sync(true) ; // upload
			//
			return *this;
		}

		~GPUstream() { 
			if (_CUDA_STREAM != NULL)
				gpuErrchk(cudaStreamDestroy(_CUDA_STREAM));
		}

		void CopyIn(T* src) {
			GPUpinned<T>::_lock(); 
			_cu_async_mem_copy<T>(src, GPUpinned<T>::_MEM, GPUpinned<T>::_SIZE, cudaMemcpyHostToHost, _CUDA_STREAM); 
			GPUpinned<T>::_unlock();
			sync(true); // upload
		}

		void CopyOut(T* dst) {
			GPUpinned<T>::_lock();
			_cu_async_mem_copy<T>(GPUpinned<T>::_MEM, dst, GPUpinned<T>::_SIZE, cudaMemcpyHostToHost, _CUDA_STREAM);
			GPUpinned<T>::_unlock();
		}

		void cuCopyIn(T* src) {
			GPUpinned<T>::_lock(); 
			_cu_async_mem_copy<T> (src, GPUpinned<T>::_MEM_GPU, GPUpinned<T>::_SIZE, cudaMemcpyDeviceToDevice, _CUDA_STREAM); 
			GPUpinned<T>::_unlock();
		}

		void cuCopyOut(T* dst) {
			GPUpinned<T>::_lock();
			_cu_async_mem_copy<T> (GPUpinned<T>::_MEM_GPU, dst, GPUpinned<T>::_SIZE, cudaMemcpyDeviceToDevice, _CUDA_STREAM); 
			GPUpinned<T>::_unlock();
		}

		void sync(bool up = false) {
			gpuErrchk(cudaStreamSynchronize(_CUDA_STREAM) );
			if (up) { _upload(); }
			else { _download(); }
		}

		// GOAL : expose  _MEM_GPU  and  _SIZE  to external cuda_kernel(...)  
		template <typename... Arguments>
		void execute_kernel(T* DST, 
							const ExecutionPolicy& ExPol, 
							void (*cuda_kernel)(T*, T*, int, Arguments...), 
							Arguments... args ) {
			/* references:
			 https://eli.thegreenplace.net/2014/variadic-templates-in-c/
			 https://en.cppreference.com/w/cpp/language/parameter_pack
			 https://developer.nvidia.com/blog/cplusplus-11-in-cuda-variadic-templates/
			 https://stackoverflow.com/questions/65886397/c-function-with-variable-number-and-types-of-arguments-as-argument-of-another
			
			// convention for cuda_kernel(DST, src, mem_index_size, other arguments...) argument list
			*/

			// std::cout << ExPol.gridSize.x << " " << ExPol.blockSize.x << std::endl;
			cuda_kernel<<<ExPol.gridSize, ExPol.blockSize, 0, _CUDA_STREAM>>>(DST, GPUpinned<T>::_MEM_GPU, GPUpinned<T>::_SIZE, args...);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				println("    error in cuda_kernel() ... ");
				std::cerr << cudaGetErrorString(err) << std::endl;
				exit(1);
			}
		}

		// special version:
		// only if cuda_kernel intends to write out to TMP 
		template <typename... Arguments>
		void execute_kernel(const ExecutionPolicy& ExPol, 
							void (*cuda_kernel)(T*, T*, int, Arguments...), 
							Arguments... args ) {
			T* TMP = _cu_async_mem_alloc_char<T>(GPUpinned<T>::_SIZE, _CUDA_STREAM) ;
			execute_kernel(TMP, ExecutionPolicy(), cuda_kernel, args...);
			GPUpinned<T>::cuCopyIn(TMP);
			GPUpinned<T>::sync(false); // download
			_cu_async_mem_free_char<T>(TMP, _CUDA_STREAM);
		}


	protected:

		cudaStream_t _CUDA_STREAM = NULL;

		void _download() { 
			GPUpinned<T>::_lock(); 
			_cu_async_mem_copy<T>(GPUpinned<T>::_MEM_GPU, GPUpinned<T>::_MEM, GPUpinned<T>::_SIZE, cudaMemcpyDeviceToHost, _CUDA_STREAM); 
			GPUpinned<T>::_unlock();
		}

		void _upload() { 
			GPUpinned<T>::_lock(); 
			_cu_async_mem_copy<T>(GPUpinned<T>::_MEM, GPUpinned<T>::_MEM_GPU, GPUpinned<T>::_SIZE, cudaMemcpyHostToDevice, _CUDA_STREAM);  
			GPUpinned<T>::_unlock();
		}

};
