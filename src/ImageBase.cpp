#pragma once

#include "ImageBase.hpp"


template <int _NCH, class M>
ImageBase<_NCH,M>::ImageBase(int H, int W, int header_size) : 
	_H(H),
	_W(W),
	_SIZE_IMG(W*H*_NCH),
	_SIZE_HEADER(header_size),
	_IMG(NULL),
	_HEADER(NULL), 
	__content_loaded(false),
	__in_operation(false)
{
	_IMG = new M(_SIZE_IMG);
	if (_SIZE_HEADER > 0)
		_HEADER = new CPU(_SIZE_HEADER);
}

template <int _NCH, class M>
ImageBase<_NCH,M>::ImageBase(ImageBase<_NCH,M> const& Img) : 
	_H(Img.get_H()),
	_W(Img.get_W()),
	_SIZE_IMG(Img.get_IMAGE_SIZE()),
	_SIZE_HEADER(Img.get_HEADER_SIZE()),
	__content_loaded(false),
	__in_operation(false)
{
	if (_IMG) { delete _IMG; }
	if (_SIZE_HEADER) { delete _HEADER; }
	_IMG = new M(*(Img._IMG));
	if (_SIZE_HEADER > 0)
		_HEADER = new CPU<char>(*(Img._HEADER));
	__content_loaded = true;
}


template <int _NCH, class M>
ImageBase<_NCH,M>& ImageBase<_NCH,M>::operator=(ImageBase<_NCH,M> const& Img)
{
	if (this == &Img) { return *this; }
	// the copy constructor repeats here
	if (_IMG) { delete _IMG; }
	if (_HEADER) { delete _HEADER; }
	_H = (Img.get_H());
	_W = (Img.get_W());
	_SIZE_IMG = (Img.get_IMAGE_SIZE());
	_SIZE_HEADER = (Img.get_HEADER_SIZE());
	__content_loaded = (false);
	__in_operation = (false);
	_IMG = new M(*(Img._IMG));
	if (_SIZE_HEADER > 0)
		_HEADER = new CPU<char>(*(Img._HEADER));
	__content_loaded = true;
	//
	return *this;
}


template <int _NCH, class M>
template <int _NCH1, class M1> 
bool ImageBase<_NCH,M>::compare_to(ImageBase<_NCH1,M1> &Img)
{
	if (_NCH1 != _NCH || Img.get_IMAGE_SIZE() != get_IMAGE_SIZE())
		return false;

	char *TMP = _mem_alloc_char<char>(Img.get_IMAGE_SIZE());
	Img.sync(false); // download
	Img.copy_out(TMP);
	for (int p=0; p<_SIZE_IMG; p++)
	{
		if (_IMG->get_byte_unsafe(p) != TMP[p])
		{
			_mem_free_char<char>(TMP);
			return false;
		}
	}
	_mem_free_char<char>(TMP);
	return true;
}



template <int _NCH, class M>
template <class MX>
void ImageBase<_NCH,M>::convert_from_mono(ImageBase<1,MX> & MonoImg, int to_channel)
{
	assert ((to_channel>=0 && to_channel<_NCH)) ;
	assert ((MonoImg.get_H()==get_H() && MonoImg.get_W()==get_W())) ;

	char *A1 = _mem_alloc_char<char>(MonoImg.get_IMAGE_SIZE());
	MonoImg.copy_out(A1);
	char *AN = _mem_alloc_char<char>(get_IMAGE_SIZE());

	for (int p=0; p<MonoImg.get_IMAGE_SIZE(); p++)
		AN[p*_NCH+to_channel]=A1[p];

	_lock(); 
	copy_in(AN); 
	sync(true);
	_unlock();
	_mem_free_char<char>(A1); 
	_mem_free_char<char>(AN);
	__content_loaded = true;
}

template <int _NCH, class M>
template <class MX>
void ImageBase<_NCH,M>::convert_from_mono_to_all_channels(ImageBase<1,MX> & MonoImg)
{
	assert ((MonoImg.get_H()==get_H() && MonoImg.get_W()==get_W())) ;

	char *A1 = _mem_alloc_char<char>(MonoImg.get_IMAGE_SIZE());
	MonoImg.copy_out(A1);
	char *AN = _mem_alloc_char<char>(get_IMAGE_SIZE());
	
	for (int to_channel=0; to_channel<_NCH; to_channel++)
		for (int p=0; p<MonoImg.get_IMAGE_SIZE(); p++)
			AN[p*_NCH+to_channel]=A1[p];

	_lock(); 
	copy_in(AN); 
	sync(true); 
	_unlock();
	_mem_free_char<char>(A1); 
	_mem_free_char<char>(AN);
	__content_loaded = true;
}

template <int _NCH, class M>
void ImageBase<_NCH,M>::to_bmp(const std::string & outfile)
{
	assert (__content_loaded);
	auto fig = BMP(_W, _H);
	char * tmp = _mem_alloc_char<char>(_SIZE_IMG);
	_IMG->copy_out(tmp); // assuming synchronized
	_lock();
	fig.RGB_interleaving_fill_data<char>(tmp, _NCH);
	_unlock();
	_mem_free_char<char>(tmp);
	fig.write(outfile.c_str());
}

template <int _NCH, class M>
void ImageBase<_NCH,M>::to_bmp_one_channel(const std::string & outfile, int ch)
{
	assert (__content_loaded);
	if ((ch < 0) || (ch > 2)) { return; } // 3 channels at most
	auto fig = BMP(_W, _H);
	char * tmp = _mem_alloc_char<char>(_SIZE_IMG);
	_IMG->copy_out(tmp); // assuming synchronized
	_lock();
	fig.fill_one_channel_interleaving<char>(tmp, ch, _NCH, 2-ch);
	_unlock();
	_mem_free_char<char>(tmp);
	fig.write(outfile.c_str()) ;
}


template <int _NCH, class M>
template <typename INTERPRETE_BYTE_AS> 
void ImageBase<_NCH,M>::to_text_file(
	const std::string & outfile, 
	const std::string & header_line
)
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
	std::ofstream fstrm(outfile.c_str(), std::ios::trunc);
	if (!fstrm)
	{
		errorln("Cannot create file! file name = ", outfile);
		fstrm.close() ;
		return;
	}
	fstrm << header_line << std::endl;
	fstrm << get_H() << " " << get_W() << " " << _NCH << std::endl;
	for (int ich=0; ich<_NCH; ich++)
	{
		fstrm << ich << std::endl;
		for (int y=0; y<_H; y++)
		{
			for (int x=0; x<_W; x++)
			{
				fstrm << static_cast<int>(static_cast<INTERPRETE_BYTE_AS>(get_pixel(x, y, ich))) << " ";
			}
			fstrm << std::endl;
		}
	}
	fstrm.close();
	return;
}
