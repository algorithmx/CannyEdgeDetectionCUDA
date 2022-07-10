# include "VisibleHumanProject.hpp"
#include "println.hpp"

const int HEIGHT = 2700 ;
const int WIDTH  = 4096 ;

void test1(void) { 
    // prepare Mempool and Stream 
    cudaMemPool_t mempool;
    gpuErrchk ( cudaDeviceGetDefaultMemPool(&mempool, 0) ) ; // device = 0
    println("GPU :");
    auto F1 = new GPU<char>(1000) ;
    println("GPU pinned :");
    auto F1a = new GPUpinned<char>(1000) ;
    println("CPU pageable :");
    auto F2 = new CPU<char>(1000) ;
    println("CPU pinned :");
    auto F3 = new CPUpinned<char>(1000) ;
    println("GPU stream :");
    auto F4 = new GPUstream<char>(1000) ;
    delete F1;
    delete F1a;
    delete F2;
    delete F3;
    delete F4;
} ;


void test_RGB_GPU_stream(int N = 12) {
    RGB_GZ_VHP<GPUstream<char>> *fig1  = new RGB_GZ_VHP<GPUstream<char>>( HEIGHT, WIDTH ) ; 
    RGB_GZ_VHP<GPUstream<char>> *fig2  = new RGB_GZ_VHP<GPUstream<char>>( HEIGHT, WIDTH ) ; 
    RGB_GZ_VHP<GPUstream<char>> *fig3  = new RGB_GZ_VHP<GPUstream<char>>( HEIGHT, WIDTH ) ; 
    RGB_GZ_VHP<GPUstream<char>> *fig4  = new RGB_GZ_VHP<GPUstream<char>>( HEIGHT, WIDTH ) ; 
    std::vector<RGB_GZ_VHP<GPUstream<char>> *> figs = {fig1, fig2, fig3, fig4} ;
    for (int i=11; i<=N; ++i) {
        auto T0 = bch::high_resolution_clock::now();
        auto FN = std::string("../data/10") + std::to_string(i) + std::string(".rgb.gz") ;
        std::cout << "processing " << FN << std::endl; 
        figs[i%4]->load_from_file( FN ) ;
        figs[i%4]->to_bmp(FN + std::string(".RGB.bmp")) ;
        auto Tf = timing(T0, "    Timing : total ") ;
    }
    delete fig1; delete fig2; delete fig3; delete fig4;
    return ;
};


void test_IO(void) {
    auto BX = GaussianFilter<3>(10,3.0);
    BX.to_text_file<unsigned char>(std::string("GaussianFilter.uchar.fltr"), std::string("gaussian"));
    auto FX = FilterFromFile<3>("GaussianFilter.uchar.fltr");
    assert ( FX.compare_to(BX) ) ;
    println("*** test_IO() is successful.");
};


void test_gaussian_filter_CPU(int N = 12) {
    //auto BX = BoxFilter<3>(5);
    auto BX = GaussianFilter<3>(10,3.0);
    BX.to_text_file<unsigned char>(std::string("box.uchar.fltr"), std::string("box"));
    BX.to_text_file<char>(std::string("box.char.fltr"), std::string("box"));
    BX.to_bmp_one_channel(std::string("box.R.bmp"), 0) ;
    BX.to_bmp_one_channel(std::string("box.G.bmp"), 1) ;
    BX.to_bmp_one_channel(std::string("box.B.bmp"), 2) ;

    RGB_GZ_VHP<CPU<char>> *fig1  = new RGB_GZ_VHP<CPU<char>>( HEIGHT, WIDTH ) ; 
    for (int i=11; i<=N; ++i) {
        auto T0 = bch::high_resolution_clock::now();
        auto FN = std::string("../data/10") + std::to_string(i) + std::string(".rgb.gz") ;
        std::cout << "processing " << FN << std::endl; 
        fig1->load_from_file( FN ) ;
        println("launching apply_filter_CPU() ... ");
        fig1->apply_filter_CPU( BX ) ;
        fig1->to_bmp(FN + std::string(".bmp")) ;
        fig1->to_bmp_one_channel(FN + std::string(".R.bmp"), 0) ;
        fig1->to_bmp_one_channel(FN + std::string(".G.bmp"), 1) ;
        fig1->to_bmp_one_channel(FN + std::string(".B.bmp"), 2) ;
        auto Tf = timing(T0, "    Timing : total ") ;
    }
    delete fig1; 
    return ;
};


void test_gaussian_filter_GPU(int N = 12) {
    auto GX = GaussianFilter<3>(101,60.0);
    RGB_GZ_VHP<GPUstream<char>> *fig1  = new RGB_GZ_VHP<GPUstream<char>>( HEIGHT, WIDTH ) ; 
    auto fig1_c = *fig1 ;
    for (int i=11; i<=N; ++i) {
        auto T0 = bch::high_resolution_clock::now();
        auto FN = std::string("../data/10") + std::to_string(i) + std::string(".rgb.gz") ;
        std::cout << "processing " << FN << std::endl; 
        fig1->load_from_file( FN ) ;
        println("launching apply_filter_GPU() ... ");
        fig1->apply_filter_GPU<unsigned char>( GX, ExecutionPolicy(dim3(16,16,4),dim3((HEIGHT+15)/16,(WIDTH+15)/16,1)) ) ;
        fig1->to_bmp(FN + std::string(".bmp")) ;
        fig1->to_bmp_one_channel(FN + std::string(".R.bmp"), 0) ;
        fig1->to_bmp_one_channel(FN + std::string(".G.bmp"), 1) ;
        fig1->to_bmp_one_channel(FN + std::string(".B.bmp"), 2) ;
        auto Tf = timing(T0, "    Timing : total ") ;
    }
    delete fig1; 
    return ;
};


void test_cast(void) {
    for (int i=-128; i<=127; i++){
        std::cout << "char_uchar_int(int_char(i=" << i << ")) = " << char_uchar_int(int_char(i)) << std::endl;
    }
};


void test_sobel_GPU(int N=15) {

    GaussianFilter<3> GF(17,5.0);
    FilterBase<3> SBX(3);
    SobelFilterX SX;
    SBX.mono2all((FilterBase<1>&)SX);
    FilterBase<3> SBY(3);
    SobelFilterY SY;
    SBY.mono2all((FilterBase<1>&)SY);

    RGB_GZ_VHP<GPU<char>> *f_Ix_amp = new RGB_GZ_VHP<GPU<char>>( HEIGHT, WIDTH ) ; 
    RGB_GZ_VHP<GPU<char>> *f_Iy = new RGB_GZ_VHP<GPU<char>>(*f_Ix_amp);
    RGB_GZ_VHP<GPU<char>> *f_dir = new RGB_GZ_VHP<GPU<char>>(*f_Ix_amp);

    auto ExPola = ExecutionPolicy(dim3(16,16,3),dim3((HEIGHT+15)/16,(WIDTH+15)/16,1));
    auto ExPolb = ExecutionPolicy(dim3(32,32,1),dim3((HEIGHT+31)/32,(WIDTH+31)/32,1));

    for (int i=11; i<=N; ++i) {

        auto T0 = bch::high_resolution_clock::now();

        auto FN = std::string("../data/10") + std::to_string(i) + std::string(".rgb.gz") ;
        std::cout << "processing " << FN << std::endl; 
        f_Ix_amp->load_from_file( FN ) ;
        *f_Iy = *f_Ix_amp;

        f_Ix_amp->apply_filter_GPU<unsigned char>( GF,  ExPola );
        // f_Ix_amp->to_bmp(FN + std::string(".gaussian1.bmp")) ;
        f_Ix_amp->apply_filter_GPU<char>( SBX, ExPola );
        // f_Ix_amp->to_bmp(FN + std::string(".sb1.bmp")) ;
        *f_dir = *f_Ix_amp;

        f_Iy->apply_filter_GPU<unsigned char>( GF,  ExPola );
        // f_Iy->to_bmp(FN + std::string(".gaussian2.bmp")) ;
        f_Iy->apply_filter_GPU<char>( SBY, ExPola );
        // f_Iy->to_bmp(FN + std::string(".sb2.bmp")) ;

        f_Ix_amp->merge_with_GPU(*f_Iy, 0, ExPola) ;
        // f_Ix_amp->to_bmp(FN + std::string(".amp.bmp")) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".R.amp.bmp"), 0) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".G.amp.bmp"), 1) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".B.amp.bmp"), 2) ;

        f_dir->merge_with_GPU(*f_Iy, 1, ExPola) ;
        // f_dir->to_bmp(FN + std::string(".dir.bmp")) ;
        // f_dir->to_bmp_one_channel(FN + std::string(".R.dir.bmp"), 0) ;
        // f_dir->to_bmp_one_channel(FN + std::string(".G.dir.bmp"), 1) ;
        // f_dir->to_bmp_one_channel(FN + std::string(".B.dir.bmp"), 2) ;

        f_Ix_amp->merge_with_GPU(*f_dir, 2, ExPola) ;
        // f_Ix_amp->to_bmp(FN + std::string(".supp.bmp")) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".R.supp.bmp"), 0) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".G.supp.bmp"), 1) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".B.supp.bmp"), 2) ;

        f_Ix_amp->merge_with_GPU(*f_Iy, 4, ExPola) ;
        f_Ix_amp->to_bmp(FN + std::string(".rescale.bmp")) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".R.rescale.bmp"), 0) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".G.rescale.bmp"), 1) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".B.rescale.bmp"), 2) ;

        f_Ix_amp->merge_with_GPU(*f_Iy, 3, ExPola) ;
        f_Ix_amp->to_bmp(FN + std::string(".DTH.bmp")) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".R.rescale.bmp"), 0) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".G.rescale.bmp"), 1) ;
        // f_Ix_amp->to_bmp_one_channel(FN + std::string(".B.rescale.bmp"), 2) ;

        auto Tf = timing(T0, "    Timing : total ") ;

    }

    delete f_Ix_amp; delete f_Iy; delete f_dir; 
    return ;
};


void test_sobel_CPU(int N=15) {

    GaussianFilter<3> GF(9,1.5);
    FilterBase<3> SBX(3);
    SobelFilterX SX;
    SBX.mono2all((FilterBase<1>&)SX);
    FilterBase<3> SBY(3);
    SobelFilterY SY;
    SBY.mono2all((FilterBase<1>&)SY);

    RGB_GZ_VHP<CPU<char>> *fig1 = new RGB_GZ_VHP<CPU<char>>( HEIGHT, WIDTH ) ; 
    RGB_GZ_VHP<CPU<char>> *fig2 = new RGB_GZ_VHP<CPU<char>>(*fig1);
    RGB_GZ_VHP<CPU<char>> *fig3 = new RGB_GZ_VHP<CPU<char>>(*fig1);
    auto ExPol = ExecutionPolicy(dim3(16,16,4),dim3((HEIGHT+15)/16,(WIDTH+15)/16,1));

    for (int i=11; i<=N; ++i) {

        auto T0 = bch::high_resolution_clock::now();

        auto FN = std::string("../data/10") + std::to_string(i) + std::string(".rgb.gz") ;
        std::cout << "processing " << FN << std::endl; 
        fig1->load_from_file( FN ) ;
        *fig2 = *fig1;

        fig1->apply_filter_CPU( GF );
        fig1->to_bmp(FN + std::string(".gaussian1.bmp")) ;
        fig1->apply_filter_CPU( SBX );
        fig1->to_bmp(FN + std::string(".sb1.bmp")) ;
        *fig3 = *fig1;

        fig2->apply_filter_CPU( GF );
        fig2->to_bmp(FN + std::string(".gaussian2.bmp")) ;
        fig2->apply_filter_CPU( SBY );
        fig2->to_bmp(FN + std::string(".sb2.bmp")) ;

        fig1->merge_with_CPU(*fig2, rgb_sobel_amp) ;
        fig1->to_bmp(FN + std::string(".amp.bmp")) ;

        fig1->to_bmp_one_channel(FN + std::string(".R.amp.bmp"), 0) ;
        fig1->to_bmp_one_channel(FN + std::string(".G.amp.bmp"), 1) ;
        fig1->to_bmp_one_channel(FN + std::string(".B.amp.bmp"), 2) ;

        fig3->merge_with_CPU(*fig2, rgb_sobel_dir) ;
        fig3->to_bmp(FN + std::string(".dir.bmp")) ;

        fig3->to_bmp_one_channel(FN + std::string(".R.dir.bmp"), 0) ;
        fig3->to_bmp_one_channel(FN + std::string(".G.dir.bmp"), 1) ;
        fig3->to_bmp_one_channel(FN + std::string(".B.dir.bmp"), 2) ;

        auto Tf = timing(T0, "    Timing : total ") ;

    }

    delete fig1; 
    delete fig2; 
    delete fig3; 
    
    return ;
};


void test_copy(void) {
    auto GF1 = FilterFromFile<1>("Gaussian.Filter");
    auto GF3red = FilterBase<3>(GF1.get_H());
    GF3red.mono2red(GF1);
    auto GF3blue = FilterBase<3>(GF1.get_H());
    GF3blue.mono2blue(GF1);
    auto GF3green = FilterBase<3>(GF1.get_H());
    GF3green.mono2green(GF1);

    GF3red.to_text_file<  unsigned char>(std::string("GaussianFilter.R.fltr"), std::string("gaussian_red"));
    GF3red.to_bmp(std::string("GaussianFilter.R.bmp")) ;
    GF3green.to_text_file<unsigned char>(std::string("GaussianFilter.G.fltr"), std::string("gaussian_green"));
    GF3green.to_bmp(std::string("GaussianFilter.G.bmp")) ;
    GF3blue.to_text_file< unsigned char>(std::string("GaussianFilter.B.fltr"), std::string("gaussian_blue"));
    GF3blue.to_bmp(std::string("GaussianFilter.B.bmp")) ;
};


int main(int argc, char** argv) {
    // test_RGB_GPU_stream() ;
    // test1() ;
    // test_gaussian_filter_CPU() ;
    // test_gaussian_filter_GPU() ;
    // test_cast();
    // test_IO();
    //test_sobel_CPU();
    test_sobel_GPU(99);
    return 0;
};
