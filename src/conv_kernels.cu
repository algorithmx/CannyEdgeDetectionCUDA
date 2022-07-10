/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *  GNU GENERAL PUBLIC LICENSE
 *  Version 3, 29 June 2007
 *
 *  Author: Yunlong Lian @algorithmx 
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#define max2(a, b)  ((a>b) ? (a) : (b))
#define min2(a, b)  ((a<b) ? (a) : (b))
#define sqr(a)      ((a)*(a))
#define iabs(a)     ((a<0) ? (-a) : a)
#define cabs(a)     ((a<0) ? (-a) : a)
#define sqf(a)      (static_cast<float>(sqr(static_cast<int>(a))))

#include <cmath>
#include "cast.hpp"

///////////////////////////////////////////////////////////////////
// CONVENTIONS: 
// (1) cuda kernel argument list
// cuda_kernel( tmp/external memory, dst/internal memory, 
//				memory index size, other arguments...) 
// (2) interleaving multichannel image linear index for a pixel (X,Y,channel)
// (channel + N_channels * (Y * Width + X))
// this is consistent with the defintion of function 
// __interleaving_multichannel_array_index_x_width_fast__()
///////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////
// FILTER CPU CODE

__host__  void conv_N_channel_interleaving_to_N_channel_filter_CPU_all_channels(
	char *image_dst,    // interleaving multi-channel img after applying fltr 
	char *image_src,    // interleaving multi-channel img before applying fltr
	int mem_index_size, // linear size of image (as char[])
	int Nchannels,      // number of channels
	char *filter,       // filter (as char[])
	int *ftot,          // normalization of filter (as int[Nchannels])
	int Hi, int Wi,     // size of image
	int Hf, int Wf      // size of filter
	) {

	int cx = (Wf - 1) / 2, cy = (Hf - 1) / 2 ; // filter centers
	int c_N = (cy * Wi + cx) * Nchannels ;     // tmp variable for speed
	int sum, maxk_N, mink_N, maxl_N, minl_N;   // tmp vars
	int k__x__Wf_N, i__x__Wi;                  // tmp vars
	int img_linear_pos_N;                      // pixel linear index * Nchannels
	int p;

	for (int i = 0; i < Hi; i++)
	{
		i__x__Wi = i * Wi;
		for (int j = 0; j < Wi; j++)
		{
			///////////////////////
			//  PIXEL OPERATION  //
			///////////////////////
			img_linear_pos_N = (i__x__Wi + j) * Nchannels;

			// boundaries of x, y coordinates of the filter
			maxk_N = (Hf - max2(0, (Hf-cy)-(Hi-i))) * Nchannels;
			mink_N = max2(0, cy-i) * Nchannels;
			maxl_N = (Wf - max2(0, (Wf-cx)-(Wi-j))) * Nchannels;
			minl_N = max2(0, cx-j) * Nchannels;

			// apply filter channelwise
			for (int ich = 0; ich < Nchannels; ich++)
			{
				p = img_linear_pos_N + ich;

				// weighted average over all filter elements
				// zero-padding, same-size
				sum = 0;
				for (int k_N = mink_N; k_N < maxk_N; k_N += Nchannels)
				{
					k__x__Wf_N = k_N * Wf ;
					for (int l_N = minl_N; l_N < maxl_N; l_N += Nchannels)
					{
						sum += (  
							static_cast<int>(filter[(k__x__Wf_N + l_N) + ich]) \
						  * char_uchar_int(image_src[p + (k_N * Wi + l_N) - c_N])
						);
					}
				}
				image_dst[p] = int_uchar_char(iabs(sum)/ftot[ich]);
			}
		}
	}
}

///////////////////////////////////////////////////////////////////
// FILTER GPU CODE

template <typename INTERPRETE_AS>
__global__  void conv_N_channel_interleaving_to_N_channel_filter_GPU_all_channels(
	char *image_dst,    // interleaving multi-channel image after applying filter 
	char *image_src,    // interleaving multi-channel image before applying filter
	int mem_index_size, // linear size of image (as char[])
	int Nchannels,      // number of channels
	char *filter,       // filter (as char[])
	int *ftot,          // normalization of filter (as int[Nchannels])
	int Hi, int Wi,     // size of image
	int Hf, int Wf      // size of filter
	) {

	// zero-padding, same-size
	int ich = threadIdx.z;
	if (ich >= Nchannels) { return; }
	int i = blockDim.x * blockIdx.x + threadIdx.x ;
	if (i >= Hi) { return; }
	int j = blockDim.y * blockIdx.y + threadIdx.y ;
	if (j >= Wi) { return; }

	// pixel linear index * num_channels
	int pli = (i * Wi + j) * Nchannels + ich;
	if (pli >= mem_index_size) { return; }

	// filter centers
	int cx = (Wf-1)/2, cy = (Hf-1)/2 ;
	// int c_N = (cy * Wi + cx) * Nchannels ;
	int pli_sub_c_N = pli - (cy * Wi + cx) * Nchannels ;

	// boundaries of x, y coordinates of the filter
	int maxk_N = (Hf - max2(0, (Hf-cy)-(Hi-i))) * Nchannels;
	int mink_N = max2(0, cy-i) * Nchannels;
	int maxl_N = (Wf - max2(0, (Wf-cx)-(Wi-j))) * Nchannels;
	int minl_N = max2(0, cx-j) * Nchannels;

	///////////////////////
	//  PIXEL OPERATION  //
	///////////////////////
	// weighted average
	// over all filter elements
	// zero-padding, same-size
	int k__x__Wf_N, k__x__Wi_N;
	int sum = 0;
	for (int k_N = mink_N; k_N < maxk_N; k_N += Nchannels)
	{
		k__x__Wf_N = k_N * Wf ;
		k__x__Wi_N = k_N * Wi ;
		for (int l_N = minl_N; l_N < maxl_N; l_N += Nchannels)
		{
			sum += ( 
				static_cast<int>(    filter[ (k__x__Wf_N + l_N) + ich        ])\
			  * cu_char_uchar_int(image_src[ (k__x__Wi_N + l_N) + pli_sub_c_N])
			);
		}
	}
	// we need typename INTERPRETE_AS here to tell the function 
	// how to convert the result
	// for filters with negative numbers (such as Sobel_x, Sobel_y), 
	// INTERPRETE_AS = char
	// otherwise 
	// INTERPRETE_AS = unsigned char
	image_dst[pli] = static_cast<char>(static_cast<INTERPRETE_AS>(sum/ftot[ich]));
}

///////////////////////////////////////////////////////////////////
// SOBEL CPU CODE

typedef char (*pfunc_sobel)(char, char); 
// tried to use function pointer, GPU code not successful, CPU code ok

char rgb_sobel_amp(char Ix, char Iy)
{
	return static_cast<char>(floorf(sqrtf(0.4999f*(sqf(Ix) + sqf(Iy)))));
}

char rgb_sobel_dir(char Ix, char Iy)
{
	// 20.37183 = 128 / (2*math.pi)
	return static_cast<char>(floorf(atan2f(Iy,Ix) * 40.74367f));
}

__host__  void inplace_merge_pixelwise_CPU_all_channels(
	char *image_src,      // in-place
	char *image_dst, 
	int mem_index_size,
	int Nchannels,
	pfunc_sobel op,
	int Hi, int Wi 
)
{
	int i__x__Wi;          // tmp vars
	int img_linear_pos_N;  // pixel linear index * num_channels
	for (int i = 0; i < Hi; i++) 
	{
		i__x__Wi = i * Wi;
		for (int j = 0; j < Wi; j++) 
		{
			img_linear_pos_N = (i__x__Wi + j) * Nchannels;
			for (int ich = 0; ich < Nchannels; ich++) 
			{
				image_dst[img_linear_pos_N+ich] = \
					op(image_dst[img_linear_pos_N+ich], 
					   image_src[img_linear_pos_N+ich]);
			}
		}
	}
}

///////////////////////////////////////////////////////////////////
// SOBEL GPU CODE

// calculate sqrt(Ix^2 + Iy^2) (always non-negative)
__device__ char cu_rgb_sobel_amp(char Ix, char Iy)
{
	return cu_uint_uchar_char(__float2uint_rd(__fsqrt_rd((sqf(Ix) + sqf(Iy)))));
}

// calculate arctan(Ix, Iy) (range (-pi, pi] )
__device__ char cu_rgb_sobel_dir(char Ix, char Iy)
{
	// rescale factor 20.37183 = 128 / (2*math.pi)
	return static_cast<char>(__float2int_rd( atan2f(Iy,Ix) * 40.74367f));
}

__global__  void inplace_merge_pixelwise_GPU_all_channels_amp(
	char *image_src,    // temp copy of Iy
	char *image_dst,    // in-place, overwrite Ix
	int mem_index_size,
	int Nchannels,
	int Hi, int Wi
)
{
	int ich = threadIdx.z;
	if (ich >= Nchannels) { return; }
	int i = blockDim.x * blockIdx.x + threadIdx.x ;
	if (i >= Hi) { return; }
	int j = blockDim.y * blockIdx.y + threadIdx.y ;
	if (j >= Wi) { return; }

	// pixel linear index * num_channels
	int p = (i * Wi + j) * Nchannels + ich;
	image_dst[p] = cu_rgb_sobel_amp(image_dst[p], image_src[p]);
}

__global__  void inplace_merge_pixelwise_GPU_all_channels_dir(
	char *Iy, // temp copy
	char *Ix, // in-place, overwrite
	int mem_index_size,
	int Nchannels,
	int Hi, int Wi
)
{
	int ich = threadIdx.z;
	if (ich >= Nchannels) { return; }
	int i = blockDim.x * blockIdx.x + threadIdx.x ;
	if (i >= Hi) { return; }
	int j = blockDim.y * blockIdx.y + threadIdx.y ;
	if (j >= Wi) { return; }

	// pixel linear index * num_channels
	int p = (i * Wi + j) * Nchannels + ich;
	Ix[p] = cu_rgb_sobel_dir(Ix[p], Iy[p]);
}


// shift-x to eight neighbors
__device__ int cu_dir2dx(char direction)
{ 
	return (cabs(direction) < 48 ? +1 : (cabs(direction) >= 80 ? -1 : 0)); 
}


// shift-y to eight neighbors
__device__ int cu_dir2dy(char direction)
{
	return ((direction >= 16 && direction < 112)  \
				? +1 \
				: ((direction <= -16 && direction > -112)  \
					? -1 \
					: 0)); 
}

__global__  void inplace_merge_pixelwise_GPU_all_channels_non_maximal_suppression(
	char *dir, // atan2(Ix,Iy), in-place
	char *amp, // sqrt(Ix^2 + Iy^2)
	int mem_index_size,
	int Nchannels,
	int Hi, int Wi
)
{
	// index
	int ich = threadIdx.z;
	if (ich >= Nchannels) { return; }
	int i = blockDim.x * blockIdx.x + threadIdx.x ;
	if (i >= Hi) { return; } // exclude edge pixels
	int j = blockDim.y * blockIdx.y + threadIdx.y ;
	if (j >= Wi) { return; } // exclude edge pixels

	int  p  = (i * Wi + j) * Nchannels + ich;
	if (i==Hi-1 || i==0 || j==0 || j==Wi-1) 
	{
		amp[p] = char(0); // suppress the image border
		return ;
	}

	// pixel linear index * num_channels
	int  dx = cu_dir2dx(dir[p]);
	int  dy = cu_dir2dy(dir[p]);
	char c1 = amp[p + ( dx + Wi * dy) * Nchannels];
	char c2 = amp[p + (-dx - Wi * dy) * Nchannels];
	if (amp[p]<c1 || amp[p]<c2) // amp[p] non-maximal
		amp[p] = char(0); // suppress
}

__device__ unsigned char cu_rescale(
	unsigned char c, 
	unsigned char cmin, 
	unsigned char cmax
)
{
	return static_cast<unsigned char>(
					256u * static_cast<unsigned int>(max2(min2(cmax,c),cmin)-cmin) \
				  / static_cast<unsigned int>(cmax-cmin+1)
		   );
}

__global__  void inplace_pixelwise_GPU_all_channels_rescale(
	char *dummy, // dummy
	char *img,   // sqrt(Ix^2 + Iy^2)
	int mem_index_size,
	int Nchannels,
	unsigned char red_min,   unsigned char red_max, 
	unsigned char green_min, unsigned char green_max, 
	unsigned char blue_min,  unsigned char blue_max, 
	int Hi, int Wi
)
{
	// index
	int ich = threadIdx.z;
	if (ich >= Nchannels) { return; }
	int i = blockDim.x * blockIdx.x + threadIdx.x ;
	if (i >= Hi) { return; } // exclude edge pixels
	int j = blockDim.y * blockIdx.y + threadIdx.y ;
	if (j >= Wi) { return; } // exclude edge pixels

	int  p  = (i * Wi + j) * Nchannels + ich;
	if (ich==0) // red
	{
		img[p] = cu_rescale(static_cast<unsigned char>(img[p]), red_min, red_max);
	} 
	else 
	if (ich==1)  // green
	{
		img[p] = cu_rescale(static_cast<unsigned char>(img[p]), green_min, green_max);
	}
	else
	if (ich==2) // blue
	{
		img[p] = cu_rescale(static_cast<unsigned char>(img[p]), blue_min,  blue_max);
	}
}

__global__  void pixelwise_GPU_all_channels_Hysteresis_Thresholding(
	char *img,   //
	char *outp,  // 
	int mem_index_size,
	int Nchannels,
	unsigned char red_low,   unsigned char red_high, 
	unsigned char green_low, unsigned char green_high, 
	unsigned char blue_low,  unsigned char blue_high, 
	int Hi, int Wi
)
{
	// index
	int ich = threadIdx.z;
	if (ich >= Nchannels) { return; }
	int i = blockDim.x * blockIdx.x + threadIdx.x ;
	if (i >= Hi) { return; } // exclude edge pixels
	int j = blockDim.y * blockIdx.y + threadIdx.y ;
	if (j >= Wi) { return; } // exclude edge pixels

	int  p   = (i * Wi + j) * Nchannels + ich;
	if (i==Hi-1 || i==0 || j==0 || j==Wi-1)
	{
		outp[p] = char(0); // mute the image border  
		return ;
	}

	outp[p] = (unsigned char)(0u);
	int Nx = Nchannels;
	int Ny = Nchannels * Wi;
	char CH_MAX = cu_uint_uchar_char(255u);
	// double thresholding
	if (ich==0) // red
	{
		if (img[p] > red_high) 
		{
			// point-p is a strong point
			outp[p] = CH_MAX;
		}
		else
		if (img[p] >= red_low) 
		{
			for (int ix=-Nx; ix<=Nx; ix+=Nx)
			{
				for (int iy=-Ny; iy<=Ny; iy+=Ny)
				{
					if (iy==0 && iy==0) 
						continue;
					if (img[p+ix+iy] > red_high)
					{
						// point-p connect to strong point
						outp[p] = CH_MAX;
						break;
					}
				}
			}
		}
	}
	else
	if (ich==1)  // green
	{
		// repeats the red code
		if (img[p] > green_high)
		{
			// point-p is a strong point
			outp[p] = CH_MAX;
		}
		else
		if (img[p] >= green_low)
		{
			for (int ix=-Nx; ix<=Nx; ix+=Nx)
			{
				for (int iy=-Ny; iy<=Ny; iy+=Ny)
				{
					if (iy==0 && iy==0) 
						continue;
					if (img[p+ix+iy] > green_high)
					{
						// point-p connect to strong point
						outp[p] = CH_MAX;
						break;
					}
				}
			}
		}
	}
	else
	if (ich==2) // blue
	{
		if (img[p] > blue_high)
		{
			// point-p is a strong point
			outp[p] = CH_MAX;
		}
		else
		if (img[p] >= blue_low)
		{
			for (int ix=-Nx; ix<=Nx; ix+=Nx)
			{
				for (int iy=-Ny; iy<=Ny; iy+=Ny)
				{
					if (iy==0 && iy==0) 
						continue;
					if (img[p+ix+iy] > blue_high)
					{
						// point-p connect to strong point
						outp[p] = CH_MAX;
						break;
					}
				}
			}
		}
	}
}
