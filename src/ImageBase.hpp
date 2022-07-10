/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *  GNU GENERAL PUBLIC LICENSE
 *  Version 3, 29 June 2007
 *
 *  Author: Yunlong Lian @algorithmx 
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#include <iostream>
#include <cassert>
#include <atomic>
#include <fstream>
#include "Mem.hpp"
#include "bmp.hpp"
#include "timing.hpp"


inline int __interleaving_multichannel_array_index_x_width_fast__(
	int X, int Y, int W, int N, int c
)
{
	return (c+N*(Y*W+X));
}


// how to use the ImageBase class
// 1. Specify the memory class M, to be : CPU, CPUpinned, GPU, GPUpinned or GPUstream
// 2. Implement protected member function _load() for particular image file type
// 3. Overwrite to_pos(x0, y0, ch) if necessary
// 4. Overwrite the constructor and destructor if necessary
template <int _NCH, class M>
class ImageBase
{
	public: 

		ImageBase () = delete ; // prevent undefined behavior

		ImageBase(int H, int W, int header_size = 0);

		ImageBase(int A): ImageBase<_NCH,M>(A, A, 0){}

		ImageBase(const ImageBase<_NCH,M>& Img);

		ImageBase<_NCH,M> &operator=(ImageBase<_NCH,M> const& Img);

		virtual ~ImageBase()
		{
			delete _IMG; 
			if (_HEADER) { delete _HEADER; };
		}

		// utilities
		bool in_box(int x0, int y0, int ch) const
		{
			return (x0<_W || y0<_H || x0>=0 || y0>=0 || ch>=0 || ch<_NCH);
		}

		int  to_pos(int x0, int y0, int ch) const 
		{ 
			// for interleaving
			return __interleaving_multichannel_array_index_x_width_fast__(x0,y0,_W,_NCH,ch);
		}

		int  get_H(void) const { return _H; }

		int  get_W(void) const { return _W; }

		int  get_HEADER_SIZE(void) const { return _SIZE_HEADER; }

		int  get_IMAGE_SIZE(void) const { return _SIZE_IMG; }

		char get_header_char(int pos) const
		{
			assert (__content_loaded); 
			return _HEADER->get_byte(pos); 
		}

		char get_header_char_unsafe(int pos) const
		{
			return _HEADER->get_byte_unsafe(pos);
		}

		char get_pixel(int x0, int y0, int ch) const
		{
			assert (__content_loaded && _IMG != NULL); 
			return (in_box(x0,y0,ch) ? get_pixel_unsafe(x0, y0, ch) : int_char(0)); 
		}

		char get_pixel_unsafe(int x0, int y0, int ch) const
		{
			return _IMG->get_byte_unsafe(to_pos(x0,y0,ch));
		}

		void set_pixel(int x0, int y0, int ch, char VAL)
		{
			assert (_IMG != NULL); 
			if (in_box(x0,y0,ch))
				set_pixel_unsafe(x0, y0, ch, VAL);
		}

		void set_pixel_unsafe(int x0, int y0, int ch, char VAL)
		{
			_IMG->set_byte_unsafe(to_pos(x0,y0,ch), VAL);
		}

		void set_val(char VAL)
		{
			for (int i=0; i<_SIZE_IMG; ++i)
				_IMG->set_byte_unsafe(i,VAL); 
		}

		void CopyOut(char * dst) { _IMG->CopyOut(dst); }

		void CopyIn (char * src) { _IMG->CopyIn (src); }

		void cuCopyOut(char * dst) { _IMG->cuCopyOut(dst); }

		void cuCopyIn (char * src) { _IMG->cuCopyIn (src); }

		void sync (bool up) { _IMG->sync(up); }

		template <int _NCH1, class M1> 
		bool compare_to(ImageBase<_NCH1,M1> &Img);

		// load and convert
		template <class MX>
		void convert_from_mono(ImageBase<1,MX> & MonoImg, int to_channel);

		template <class MX>
		void convert_from_mono_to_all_channels(ImageBase<1,MX> & MonoImg);

		void load_from_file(const std::string & file_name) { _load(file_name); }

		void to_bmp(const std::string & outfile);

		void to_bmp_one_channel(const std::string & outfile, int ch = 0);

		template <typename INTERPRETE_BYTE_AS = char> 
		void to_text_file(const std::string & outfile, const std::string & header_line);


	protected:

		int _W = 0, _H = 0, _SIZE_IMG = 0, _SIZE_HEADER = 0;

		M * _IMG = NULL;

		CPU<char> * _HEADER = NULL;

		std::atomic<bool> __content_loaded = false;

		std::atomic<bool> __in_operation = false;

		void _lock(void)
		{
			while ( __in_operation )
				wait();
			__in_operation = true ; 
		}

		void _unlock(void) { __in_operation = false; }

		virtual void _load (const std::string & file_name) = 0 ;

};
