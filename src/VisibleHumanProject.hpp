/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *  GNU GENERAL PUBLIC LICENSE
 *  Version 3, 29 June 2007
 *
 *  Author: Yunlong Lian @algorithmx 
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#include <cassert>
#include <fstream>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include "println.hpp"
#include "conv_kernels.cu"
#include "ImageBase.cpp"
#include "Filter.hpp"


// Visible Human Project, CT .rgb.gz images
template <class M>
class RGB_GZ_VHP : public ImageBase<3,M>
{

    public:

        RGB_GZ_VHP() = delete ;

        RGB_GZ_VHP(int H=2700, int W=4096) : ImageBase<3,M>(H,W,0) {}

        ~RGB_GZ_VHP(){}

        void apply_filter_CPU(FilterBase<3> &F, char *DST)
        {
            int *tot = _mem_alloc_char<int>(3);
            for (int p=0; p<3; p++)
                tot[p] = F.get_total(p);

            char *A = _mem_alloc_char<char>(F.get_IMAGE_SIZE());
            F.copy_out(A);

            ImageBase<3,M>::_IMG->execute_kernel(
                DST, ExecutionPolicy(),
                &conv_N_channel_interleaving_to_N_channel_filter_CPU_all_channels,
                3, A, tot, ImageBase<3,M>::get_H(), ImageBase<3,M>::get_W(), F.get_H(), F.get_W() 
            );
    
            _mem_free_char<char>(A); 
            _mem_free_char<int>(tot); 
        }


        void apply_filter_CPU(FilterBase<3> &F) {
            char *DST = _mem_alloc_char<char>(ImageBase<3,M>::get_IMAGE_SIZE());
            apply_filter_CPU(F, DST);
            ImageBase<3,M>::copy_in(DST);
            ImageBase<3,M>::sync(true); // upload
            _mem_free_char<char>(DST); 
        }

        template <typename INTERPRETE_AS>
        void apply_filter_GPU(FilterBase<3> &F, char *DST, const ExecutionPolicy & ExPol)
        {
            int F_IMG_SIZE = F.get_IMAGE_SIZE();
            int *tot = _mem_alloc_char<int>(3);
            for (int p=0; p<3; p++) 
                tot[p]=F.get_total(p);
            int *cu_tot = _cu_mem_alloc_char<int>(3);
            _cu_mem_copy<int>(tot, cu_tot, 3, cudaMemcpyHostToDevice);

            char *A = _mem_alloc_char<char>(F_IMG_SIZE);
            F.copy_out(A) ;
            char *cu_A = _cu_mem_alloc_char<char>(F_IMG_SIZE);
            _cu_mem_copy<char>(A, cu_A, F_IMG_SIZE, cudaMemcpyHostToDevice);

            ImageBase<3,M>::_IMG->execute_kernel(
                DST, ExPol, 
                &conv_N_channel_interleaving_to_N_channel_filter_GPU_all_channels<INTERPRETE_AS>,
                3, cu_A, cu_tot, ImageBase<3,M>::get_H(), ImageBase<3,M>::get_W(), F.get_H(), F.get_W() 
            );

            _mem_free_char<char>(A); 
            _mem_free_char<int>(tot); 
            _cu_mem_free_char<char>(cu_A); 
            _cu_mem_free_char<int>(cu_tot); 
        }

        template <typename INTERPRETE_AS>
        void apply_filter_GPU(FilterBase<3> &F, const ExecutionPolicy & ExPol) {
            char *DST = _cu_mem_alloc_char<char>(ImageBase<3,M>::get_IMAGE_SIZE());
            apply_filter_GPU<INTERPRETE_AS>(F, DST, ExPol);
            ImageBase<3,M>::_IMG->cu_copy_in(DST);
            ImageBase<3,M>::sync(false); // download to CPU memory
            _cu_mem_free_char<char>(DST); 
        }


        void merge_with_CPU(RGB_GZ_VHP<M> &RGB1, char (*rgb_op)(char, char))
        {
            if (
                (RGB1.get_IMAGE_SIZE() != ImageBase<3,M>::get_IMAGE_SIZE()) ||
                (RGB1.get_H() != ImageBase<3,M>::get_H()) ||
                (RGB1.get_W() != ImageBase<3,M>::get_W())
            )
            {
                return;
            }

            char *DST = _mem_alloc_char<char>(ImageBase<3,M>::get_IMAGE_SIZE());
            RGB1.copy_out(DST); // assuming synchronized

            ImageBase<3,M>::_IMG->execute_kernel(
                DST, ExecutionPolicy(),
                &inplace_merge_pixelwise_CPU_all_channels,
                3, rgb_op, ImageBase<3,M>::get_H(), ImageBase<3,M>::get_W()
            );

            // no need to copy-in, kernel operation is in-place
            ImageBase<3,M>::sync(true); // upload 
            _mem_free_char<char>(DST); 
        }

        void merge_with_GPU(RGB_GZ_VHP<M> & RGB1, int mode_op, const ExecutionPolicy & ExPol)
        {
            if (
                (RGB1.get_IMAGE_SIZE() != ImageBase<3,M>::get_IMAGE_SIZE()) ||
                (RGB1.get_H() != ImageBase<3,M>::get_H()) ||
                (RGB1.get_W() != ImageBase<3,M>::get_W())
            )
            {
                return;
            }

            char *cu_TMP = _cu_mem_alloc_char<char>(ImageBase<3,M>::get_IMAGE_SIZE());

            if (mode_op==0)
            {
                RGB1.cu_copy_out(cu_TMP); // assuming synchronized
                ImageBase<3,M>::_IMG->execute_kernel(
                    cu_TMP, ExPol,
                    &inplace_merge_pixelwise_GPU_all_channels_amp, // sqrt(Ix^2 + Iy^2)
                    3, ImageBase<3,M>::get_H(), ImageBase<3,M>::get_W()
                );
            }
            else
            if (mode_op==1)
            {
                RGB1.cu_copy_out(cu_TMP); // assuming synchronized
                ImageBase<3,M>::_IMG->execute_kernel(
                    cu_TMP, ExPol,
                    &inplace_merge_pixelwise_GPU_all_channels_dir, // atan2(Ix, Iy)
                    3, ImageBase<3,M>::get_H(), ImageBase<3,M>::get_W()
                );
            }
            else
            if (mode_op==2)
            {
                RGB1.cu_copy_out(cu_TMP); // assuming synchronized
                ImageBase<3,M>::_IMG->execute_kernel(
                    cu_TMP, ExPol,
                    &inplace_merge_pixelwise_GPU_all_channels_non_maximal_suppression, // non_maximal_suppression
                    3, ImageBase<3,M>::get_H(), ImageBase<3,M>::get_W() );
            }
            else 
            if (mode_op==3) 
            {
                ImageBase<3,M>::cu_copy_out(cu_TMP); // assuming synchronized
                ImageBase<3,M>::_IMG->execute_kernel(
                    cu_TMP, ExPol,
                    &pixelwise_GPU_all_channels_Hysteresis_Thresholding, // Hysteresis_Thresholding
                    3, 
                    (unsigned char)8u, (unsigned char)16u, 
                    (unsigned char)8u, (unsigned char)16u, 
                    (unsigned char)8u, (unsigned char)16u, 
                    ImageBase<3,M>::get_H(), ImageBase<3,M>::get_W()
                );
            }
            else
            if (mode_op==4) 
            {
                // RGB1.cu_copy_out(cu_TMP); // cu_TMP is a dummy
                ImageBase<3,M>::_IMG->execute_kernel(
                    cu_TMP, ExPol,
                    &inplace_pixelwise_GPU_all_channels_rescale, // rescale
                    3, 
                    (unsigned char)2u, (unsigned char)64u, 
                    (unsigned char)2u, (unsigned char)64u, 
                    (unsigned char)2u, (unsigned char)64u, 
                    ImageBase<3,M>::get_H(), ImageBase<3,M>::get_W()
                );
            }

            // no need to copy-in, kernel operation is in-place
            cudaDeviceSynchronize();
            ImageBase<3,M>::sync(false); // download to CPU memory
            _cu_mem_free_char<char>(cu_TMP); 
        }


    protected:

        void _load(const std::string & file_name)
        {
            auto T0 = bch::high_resolution_clock::now();
            namespace bio = boost::iostreams;
            // load file
            std::basic_ifstream<char> fstrm( file_name.c_str(), std::ios::in | std::ios::binary );
            if (!fstrm)
            {
                errorln("File not found! file name = ", file_name);
                ImageBase<3,M>::__content_loaded = false ;
                fstrm.close() ;
                return;
            }
            // decompression
            // https://www.boost.org/doc/libs/1_49_0/libs/iostreams/doc/classes/gzip.html
            bio::filtering_streambuf<bio::input> input;
            input.push( bio::gzip_decompressor() );
            input.push( fstrm );
            std::basic_istream<char> inflated(&input);
            int size = ImageBase<3,M>::_SIZE_IMG;
            char *LOCAL_MEM = _mem_alloc_char<char>(size);
            try
            {
                inflated.read(LOCAL_MEM, (size)*sizeof(char));
            }
            catch (const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
                std::cerr << "_dcmprss(file = " << file_name << ") failed." << std::endl;
                ImageBase<3,M>::_unlock() ;
                ImageBase<3,M>::__content_loaded = false ;
                fstrm.close() ;
                return ;
            }
            ImageBase<3,M>::_lock(); 
            ImageBase<3,M>::_IMG->copy_in(LOCAL_MEM); 
            ImageBase<3,M>::_IMG->sync(true); 
            ImageBase<3,M>::_unlock();
            ImageBase<3,M>::__content_loaded = true ;
            _mem_free_char<char>(LOCAL_MEM);
            auto Tf = timing(T0, "    Timing : RGB_VHP::_load() ");
        }

};


// Visible Human Project, CT .fre and .fro images
template <class M>
class CT_VHP : public ImageBase<1,M> 
{

    public:

        CT_VHP() = delete ;

        CT_VHP(int H=512, int W=512, int hs=3416) : ImageBase<1,M>(H,W,hs) {}

        ~CT_VHP(){}

    protected:

        void _load(const std::string & file_name) 
        {
            namespace bio = boost::iostreams;
            // load file
            std::basic_ifstream<char> fstrm( file_name.c_str(), std::ios::in | std::ios::binary );
            if (!fstrm) 
            {
                ImageBase<1,M>::__content_loaded = false ;
                fstrm.close() ;
                return;
            }

            try
            {
                int size = ImageBase<1,M>::_SIZE_IMG ;
                int header_size = ImageBase<1,M>::_SIZE_HEADER ;
                char *TMP = _mem_alloc_char<char>(size*2);
                char *LOCAL_MEM = _mem_alloc_char<char>(size);
                char *LOCAL_HEADER;
                if (header_size>0) 
                {
                    LOCAL_HEADER = _mem_alloc_char<char>(header_size);
                    fstrm.read(LOCAL_HEADER, (header_size)*sizeof(char));
                }
                fstrm.read(TMP, (size)*sizeof(char));

                for (int q=0; q<size; q++)
                    LOCAL_MEM[q] = TMP[2*q + 1]; 
                    // +1 is special for the data file GE CT scan
                    // claims 16-bit image file but only half of the pixel is used

                ImageBase<1,M>::_lock(); 
                ImageBase<1,M>::copy_in(LOCAL_MEM); 
                ImageBase<1,M>::sync(true); 
                if (header_size>0) 
                    ImageBase<1,M>::_HEADER->copy_in(LOCAL_HEADER); 
                ImageBase<1,M>::_unlock();
                ImageBase<1,M>::__content_loaded = true ;

                if (header_size>0)
                    _mem_free_char<char>(LOCAL_HEADER);
                _mem_free_char<char>(LOCAL_MEM);
                _mem_free_char<char>(TMP);
            } 
            catch (const std::exception& e) 
            {
                std::cerr << e.what() << std::endl;
                std::cerr << "_dcmprss(file = " << file_name << ") failed." << std::endl;
                ImageBase<1,M>::_unlock() ;
                ImageBase<1,M>::__content_loaded = false ;
                fstrm.close() ;
                return ;
            }
        }

};
