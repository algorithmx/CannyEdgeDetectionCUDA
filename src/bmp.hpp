/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *  GNU GENERAL PUBLIC LICENSE
 *  Version 3, 29 June 2007
 *
 *  Author @algorithmx   
 *  BASED ON
 *  https://github.com/sol-prog/cpp-bmp-images/blob/master/BMP.h
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>


#pragma pack(push, 1)
struct BMPFileHeader {
	uint16_t file_type{ 0x4D42 };          // File type always BM which is 0x4D42 (stored as hex uint16_t in little endian)
	uint32_t file_size{ 0 };               // Size of the file (in bytes)
	uint16_t reserved1{ 0 };               // Reserved, always 0
	uint16_t reserved2{ 0 };               // Reserved, always 0
	uint32_t offset_data{ 0 };             // Start position of pixel data (bytes from the beginning of the file)
};

struct BMPInfoHeader {
	uint32_t size{ 0 };                      // Size of this header (in bytes)
	int32_t width{ 0 };                      // width of bitmap in pixels
	int32_t height{ 0 };                     // width of bitmap in pixels
											 //       (if positive, bottom-up, with origin in lower left corner)
											 //       (if negative, top-down, with origin in upper left corner)
	uint16_t planes{ 1 };                    // No. of planes for the target device, this is always 1
	uint16_t bit_count{ 0 };                 // No. of bits per pixel
	uint32_t compression{ 0 };               // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images
	uint32_t size_image{ 0 };                // 0 - for uncompressed images
	int32_t x_pixels_per_meter{ 0 };
	int32_t y_pixels_per_meter{ 0 };
	uint32_t colors_used{ 0 };               // No. color indexes in the color table. Use 0 for the max number of colors allowed by bit_count
	uint32_t colors_important{ 0 };          // No. of colors used for displaying the bitmap. If 0 all colors are required
};

struct BMPColorHeader {
	uint32_t red_mask{ 0x00ff0000 };         // Bit mask for the red channel
	uint32_t green_mask{ 0x0000ff00 };       // Bit mask for the green channel
	uint32_t blue_mask{ 0x000000ff };        // Bit mask for the blue channel
	uint32_t alpha_mask{ 0xff000000 };       // Bit mask for the alpha channel
	uint32_t color_space_type{ 0x73524742 }; // Default "sRGB" (0x73524742)
	uint32_t unused[16]{ 0 };                // Unused data for sRGB color space
};
#pragma pack(pop)



struct BMP {
	BMPFileHeader file_header;
	BMPInfoHeader bmp_info_header;
	BMPColorHeader bmp_color_header;
	std::vector<uint8_t> data;
	int _nch, _W,  _H;

	BMP(int32_t width, int32_t height, bool has_alpha=false) {
		bmp_info_header.width       = width;
		bmp_info_header.height      = height;
		bmp_info_header.size        = sizeof(BMPInfoHeader);
		file_header.offset_data     = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
		bmp_info_header.bit_count   = has_alpha ? (uint16_t)32 : (uint16_t)24 ;
		bmp_info_header.compression = has_alpha ? (uint32_t)3 : (uint32_t)0 ;
		row_stride = ( (has_alpha ? (uint32_t)4 : (uint32_t)3) * static_cast<uint32_t>(width) ) ;
		data.resize(static_cast<int32_t>(row_stride) * height);
		file_header.file_size = file_header.offset_data ;
		if (has_alpha) {
			bmp_info_header.size    += sizeof(BMPColorHeader);
			file_header.offset_data += sizeof(BMPColorHeader);
			file_header.file_size   += data.size();
		} else {
			file_header.file_size   += (static_cast<uint32_t>(data.size()) + bmp_info_header.height * (make_stride_aligned(4) - row_stride));
		}
		// auxiliary variables
		// number of channels
		_nch = static_cast<int>(bmp_info_header.bit_count) / 8;
		_W   = static_cast<int>(bmp_info_header.width ); 
		_H   = static_cast<int>(bmp_info_header.height);
	};

	template <typename T> 
	void fill_one_channel_interleaving(T *D, int ch, int N_interleaving_bytes, int to_ch) {
		for (int y = 0; y < _H; ++y) {
			for (int x=0; x<_W; ++x)
				data[_nch*(y*_W+x)+to_ch] = static_cast<uint8_t>( D[N_interleaving_bytes*(y*_W+x)+ch] );
		}
	};

	//  ( B, 0 ) ;
	//  ( G, 1 ) ;
	//  ( R, 2 ) ;
	template <typename T> 
	void RGB_interleaving_fill_data(T *IMG, int N_interleaving_bytes) {
		fill_one_channel_interleaving<T> ( IMG, std::min(0,N_interleaving_bytes-1), N_interleaving_bytes, 2 ) ;
		fill_one_channel_interleaving<T> ( IMG, std::min(1,N_interleaving_bytes-1), N_interleaving_bytes, 1 ) ;
		fill_one_channel_interleaving<T> ( IMG, std::min(2,N_interleaving_bytes-1), N_interleaving_bytes, 0 ) ;
	};

	void write(const char *fname) {
		std::ofstream of{ fname, std::ios_base::binary };
		if (of) {
			if (bmp_info_header.bit_count == 32) {
				write_headers_and_data(of);
			}
			else if (bmp_info_header.bit_count == 24) {
				if (bmp_info_header.width % 4 == 0) {
					write_headers_and_data(of);
				} else {
					uint32_t new_stride = make_stride_aligned(4);
					std::vector<uint8_t> padding_row(new_stride - row_stride);
					write_headers(of);
					for (int y = 0; y < bmp_info_header.height; ++y) {
						of.write((const char*)(data.data() + row_stride * y), row_stride);
						of.write((const char*)padding_row.data(), padding_row.size());
					}
				}
			}
			else { throw std::runtime_error("The program can treat only 24 or 32 bits per pixel BMP files"); }
		}
		else { throw std::runtime_error("Unable to open the output image file."); }
	};

	void set_pixel(int x0, int y0, uint8_t B, uint8_t G, uint8_t R) {
		if (x0 >= _W || y0 >= _H || x0 < 0 || y0 < 0) { return; }
		int P = _nch*(y0*_W+x0) ;
		data[P+0] = B;  data[P+1] = G;  data[P+2] = R;
	};


private:

	uint32_t row_stride{ 0 };

	void write_headers(std::ofstream &of) {
		of.write((const char*)&file_header, sizeof(file_header));
		of.write((const char*)&bmp_info_header, sizeof(bmp_info_header));
		if(bmp_info_header.bit_count == 32) {
			of.write((const char*)&bmp_color_header, sizeof(bmp_color_header));
		}
	}

	void write_headers_and_data(std::ofstream &of) {
		write_headers(of);
		of.write((const char*)data.data(), data.size());
	}

	// Add 1 to the row_stride until it is divisible with align_stride
	uint32_t make_stride_aligned(uint32_t align_stride) {
		uint32_t new_stride = row_stride;
		while (new_stride % align_stride != 0) { new_stride++; }
		return new_stride;
	}

	// Check if the pixel data is stored as BGRA and if the color space type is sRGB
	void check_color_header(BMPColorHeader &bmp_color_header) {
		BMPColorHeader expected_color_header;
		if(expected_color_header.red_mask != bmp_color_header.red_mask ||
			expected_color_header.blue_mask != bmp_color_header.blue_mask ||
			expected_color_header.green_mask != bmp_color_header.green_mask ||
			expected_color_header.alpha_mask != bmp_color_header.alpha_mask) {
			throw std::runtime_error("Unexpected color mask format! The program expects the pixel data to be in the BGRA format");
		}
		if(expected_color_header.color_space_type != bmp_color_header.color_space_type) {
			throw std::runtime_error("Unexpected color space type! The program expects sRGB values");
		}
	}

};