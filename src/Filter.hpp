/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *  GNU GENERAL PUBLIC LICENSE
 *  Version 3, 29 June 2007
 *
 *  Author: Yunlong Lian @algorithmx 
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#include <iostream>
#include <cmath>
#include "ImageBase.cpp"


inline char ratio2char(double X)
{
	return static_cast<char>(static_cast<int>(std::trunc((128.0-1e-8) * X)));
}


inline double gaussian(int x, int y, double sX, double sY, double sigma)
{
	return std::exp(-((y-sY)*(y-sY)+(x-sX)*(x-sX))/(sigma*sigma));
}

template <int _NCHF>
class FilterBase : public ImageBase<_NCHF,CPU<char>>
{
	public:

		FilterBase() = delete ;

		FilterBase (int A) : 
			ImageBase<_NCHF,CPU<char>>(A), 
			filter_type{""} 
		{
			/* do nothing */
		}

		FilterBase ( const FilterBase<_NCHF>& F ) :
			ImageBase<_NCHF,CPU<char>>(F)
		{
			filter_type = F.filter_type;
			for(int ich=0; ich<_NCHF; ++ich)
				_total[ich] = F.get_total(ich);
		}

		FilterBase<_NCHF>& operator=(FilterBase<_NCHF> const& F)
		{
			if (this == &F)
				return *this;

			// copy constructor repeats here
			if (ImageBase<_NCHF,CPU<char>>::_IMG)
				delete ImageBase<_NCHF,CPU<char>>::_IMG;
		
			if (ImageBase<_NCHF,CPU<char>>::_HEADER)
				delete ImageBase<_NCHF,CPU<char>>::_HEADER;

			ImageBase<_NCHF,CPU<char>>::_W = F.get_W();
			ImageBase<_NCHF,CPU<char>>::_H = F.get_H();
			ImageBase<_NCHF,CPU<char>>::__content_loaded = false;
			ImageBase<_NCHF,CPU<char>>::__in_operation = false;
			ImageBase<_NCHF,CPU<char>>::_SIZE_HEADER = F.get_HEADER_SIZE();
			ImageBase<_NCHF,CPU<char>>::_SIZE_IMG = F.get_IMAGE_SIZE();

			ImageBase<_NCHF,CPU<char>>::_IMG = new CPU<char>(*(F._IMG));

			if (ImageBase<_NCHF,CPU<char>>::_SIZE_HEADER > 0)
				ImageBase<_NCHF,CPU<char>>::_HEADER = new CPU<char>(*(F._HEADER));

			for(int ich=0; ich<_NCHF; ++ich)
				_total[ich] = F.get_total(ich);
			filter_type = F.filter_type;

			ImageBase<_NCHF,CPU<char>>::__content_loaded = true;

			return *this;
		}

		~FilterBase(){}

		void mono2red(FilterBase<1> & Mono)
		{ 
			ImageBase<_NCHF,CPU<char>>::convert_from_mono(Mono,0); 
			_total[0] = Mono.get_total(0);  
			_total[1] = 1; 
			_total[2] = 1;
		}

		void mono2green(FilterBase<1> & Mono)
		{ 
			ImageBase<_NCHF,CPU<char>>::convert_from_mono(Mono,1); 
			_total[0] = 1;  
			_total[1] = Mono.get_total(0); 
			_total[2] = 1;
		}

		void mono2blue (FilterBase<1> & Mono)
		{ 
			ImageBase<_NCHF,CPU<char>>::convert_from_mono(Mono,2); 
			_total[0] = 1;  
			_total[1] = 1; 
			_total[2] = Mono.get_total(0);
		}

		void mono2all  (FilterBase<1> & Mono)
		{ 
			ImageBase<_NCHF,CPU<char>>::convert_from_mono_to_all_channels(Mono); 
			init_total(Mono.get_total(0));
		}

		int  get_total(int i) const { return _total[i]; }

		std::string get_filter_type(void) const { return filter_type; }

		void init_total(int val = 0)
		{ 
			for(int ich=0; ich<_NCHF; ++ich)
				_total[ich] = val;
		}


	protected:

		std::string filter_type;

		int _total[_NCHF];

		void _load(const std::string & file_name){}

		void _set_total(void)
		{
			for (int ich=0; ich<_NCHF; ich++)
			{
				int sum = 0;
				for (int p=ich; p<ImageBase<_NCHF,CPU<char>>::_SIZE_IMG; p+=_NCHF) 
					sum += static_cast<int>(ImageBase<_NCHF,CPU<char>>::_IMG->get_byte_unsafe(p));
				_total[ich] = (sum==0 ? 1 : sum) ;
			}
		}
};



template <int _NCHF>
class FilterFromFile : public FilterBase<_NCHF>
{
	public: 

		FilterFromFile(const std::string & file_name) : 
			FilterBase<_NCHF>(0)
		{
			_load(file_name); 
		}

		~FilterFromFile(){}


	protected:

		void _load (const std::string & file_name)
		{
			/***
			 * FILE FORMAT:
			 * FILTER_NAME
			 * H  W  NCH
			 * CHANNEL_#
			 * X X X ... X
			 * X X X ... X
			 * ...
			 * X X X ... X
			 * CHANNEL_#
			 * X X X ... X
			 * X X X ... X
			 * ...
			 * X X X ... X
			 * (repeat ...)
			 */
			// load file
			std::ifstream fstrm( file_name.c_str(), std::ios::in );
			if (!fstrm)
			{
				errorln("File not found! file name = ", file_name);
				FilterBase<_NCHF>::__content_loaded = false ;
				fstrm.close();
				return;
			}

			// read in header lines (two lines)
			int _nch;
			fstrm >> FilterBase<_NCHF>::filter_type;
			fstrm >> FilterBase<_NCHF>::_H >> FilterBase<_NCHF>::_W >> _nch;
			assert ((_nch==_NCHF));
			FilterBase<_NCHF>::_SIZE_IMG = \
				_nch * FilterBase<_NCHF>::_H * FilterBase<_NCHF>::_W ;

			// alloc _IMG
			FilterBase<_NCHF>::_IMG = new CPU(FilterBase<_NCHF>::_SIZE_IMG);

			// read in blocks
			int tmp, ch;
			FilterBase<_NCHF>::_lock();
			for (int ich=0; ich<_NCHF; ich++)
			{
				fstrm >> ch;
				for (int y=0; y<FilterBase<_NCHF>::get_H(); y++)
				{
					// read in lines in a block
					for (int x=0; x<FilterBase<_NCHF>::get_W(); x++)
					{
						fstrm >> tmp; // read in int
						assert ((tmp>=-128 && tmp<=127)); // format check
						FilterBase<_NCHF>::set_pixel(x, y, ch, static_cast<char>(tmp));
					}
				}
			}
			FilterBase<_NCHF>::_unlock();

			// finishing
			fstrm.close() ;
			FilterBase<_NCHF>::_set_total();
			FilterBase<_NCHF>::__content_loaded = true;

			return;
		}
};


class SobelFilterX : public FilterBase<1> 
{
	public: 

		SobelFilterX(int normalization=4) : FilterBase<1>(3)
		{ 
			FilterBase<1>::filter_type = "sobel_x";
			FilterBase<1>::_lock(); 
			FilterBase<1>::set_pixel(0,0,0,char(-1)); 
			FilterBase<1>::set_pixel(1,0,0,char(-2)); 
			FilterBase<1>::set_pixel(2,0,0,char(-1)); 
			FilterBase<1>::set_pixel(0,1,0,char(0)); 
			FilterBase<1>::set_pixel(1,1,0,char(0)); 
			FilterBase<1>::set_pixel(2,1,0,char(0)); 
			FilterBase<1>::set_pixel(0,2,0,char(1)); 
			FilterBase<1>::set_pixel(1,2,0,char(2)); 
			FilterBase<1>::set_pixel(2,2,0,char(1)); 
			FilterBase<1>::init_total(normalization);
			FilterBase<1>::_unlock(); 
			FilterBase<1>::__content_loaded = true;
		}

		~SobelFilterX(){}

	protected:

		void _load (const std::string & file_name){}

};


class SobelFilterY : public FilterBase<1>
{
	public: 

		SobelFilterY (int normalization=4) : FilterBase<1>(3)
		{ 
			FilterBase<1>::filter_type = "sobel_x";
			FilterBase<1>::_lock(); 
			FilterBase<1>::set_pixel(0,0,0,char(-1)); 
			FilterBase<1>::set_pixel(0,1,0,char(-2)); 
			FilterBase<1>::set_pixel(0,2,0,char(-1)); 
			FilterBase<1>::set_pixel(1,0,0,char(0)); 
			FilterBase<1>::set_pixel(1,1,0,char(0)); 
			FilterBase<1>::set_pixel(1,2,0,char(0)); 
			FilterBase<1>::set_pixel(2,0,0,char(1)); 
			FilterBase<1>::set_pixel(2,1,0,char(2)); 
			FilterBase<1>::set_pixel(2,2,0,char(1)); 
			FilterBase<1>::init_total(normalization);
			FilterBase<1>::_unlock(); 
			FilterBase<1>::__content_loaded = true;
		}

		~SobelFilterY(){}

	protected:

		void _load (const std::string & file_name){}

};


template <int _NCHF>
class BoxFilter : public FilterBase<_NCHF> 
{
	public: 

		BoxFilter(int A) : FilterBase<_NCHF>(A)
		{ 
			FilterBase<_NCHF>::filter_type = "box";
			FilterBase<_NCHF>::init_total(A*A);
			FilterBase<_NCHF>::_lock(); 
			FilterBase<_NCHF>::set_val(static_cast<char>(1)); 
			FilterBase<_NCHF>::_unlock(); 
			FilterBase<_NCHF>::__content_loaded = true;
		}

		~BoxFilter(){}

	protected:

		void _load(const std::string & file_name){}

};


class BoxFilterOneColor : public FilterBase<3>
{
	public: 

		BoxFilterOneColor(int A, char ch) : FilterBase<3>(A)
		{
			assert ((ch=='r' || ch=='g' || ch=='b'));
			_CH = (ch=='r' ? 0 : (ch=='g' ? 1 : 2)) ;
			FilterBase<3>::filter_type = \
				(ch=='r' ? "box_red" : (ch=='g' ? "box_green" : "box_blue"));

			char c1 = static_cast<char>(1);
			FilterBase<3>::_lock(); 
			for (int i=0; i<A; ++i)
			{
				for (int j=0; j<A; ++j)
				{
					set_pixel_unsafe(i,j,_CH,c1);
				}
			}
			FilterBase<3>::_unlock();

			FilterBase<3>::_set_total();
			FilterBase<3>::__content_loaded = true;
		}

		~BoxFilterOneColor(){}

	protected:

		int _CH; 

		void _load(const std::string & file_name){}

};



template <int _NCHF>
class GaussianFilter : public FilterBase<_NCHF>
{
	public: 

		GaussianFilter(int A, double sigma) : 
			FilterBase<_NCHF>(A), _sigma(sigma)
		{
				FilterBase<_NCHF>::filter_type = "gaussian";
				FilterBase<_NCHF>::_lock(); 
				set_normalized_gaussian(sigma); 
				FilterBase<_NCHF>::_unlock(); 
				FilterBase<_NCHF>::__content_loaded = true;
		};

		~GaussianFilter(){};

		void set_normalized_gaussian(double s)
		{
			_sigma = s;
			int dimW = FilterBase<_NCHF>::get_W();
			int dimH = FilterBase<_NCHF>::get_H();
			// meaning of sX , sY
			// dim=5 : [0,1,2,3,4] => [-2,-1,0,1,2] => s_ = 2
			// dim=6 : [0,1,2,3,4,5] => [-2.5,-1.5,-0.5,0.5,1.5,2.5] => s_ = 2.5
			double sX = (dimW-1)/2.0,  sY = (dimH-1)/2.0 ;
			// malloc
			double **F = new double*[dimH] ;
			for (int i=0;i<(dimH);i++)  
				F[i] = new double[dimW] ;
			// fill the double array F[][]
			double max_elem = 0.0;
			for (int y=0; y < dimH; y++)
			{
				for (int x=0; x < dimW; x++)
				{
					// math :  F[y][x] = std::exp( - ((y-sY)^2 + (x-sX)^2)/(_sigma^2) ) ;
					F[y][x] = gaussian(x, y, sX, sY, _sigma);
					max_elem = std::max(max_elem, F[y][x]);
				}
			}
			// round and copy, all channels have equal values
			max_elem += 1e-8 ;
			for (int y=0; y < dimH; y++)
			{
				for (int x=0; x < dimW; x++)
				{
					for (int ch=0; ch < _NCHF; ch++)
						FilterBase<_NCHF>::set_pixel_unsafe(x,y,ch,ratio2char(F[y][x]/max_elem));
				}
			}
			// clean up
			for (int i=0;i<(dimH);i++)  
				delete[] F[i] ;
			delete[] F ;
			// set total
			FilterBase<_NCHF>::_set_total();
			return ;
		};

	protected:

		double _sigma ;

		void _load(const std::string & file_name){};

};

