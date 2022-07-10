#include <tuple>
#include "VisibleHumanProject.hpp"
#include "println.hpp"

// Fixed for Visible Human Project
const int HEIGHT = 2700 ;
const int WIDTH  = 4096 ;

void CannyStepVHP(
    const std::string & FN, 
    bool is_step_mode = true,
    int gaussian_filter_size = 11, 
    double gaussian_filter_sigma = 5.0,
    unsigned int cutoff = 64u,
    unsigned int thrL = 8u,
    unsigned int thrH = 16u
)
{
    std::cout << "\nCannyStepVHP() started with file " << FN \
              << "\n    Gaussian filter size = " << gaussian_filter_size \
              << " , sigma = " << gaussian_filter_sigma \
              << "\n    Rescaling cutoff = " << cutoff\
              << "\n    Double threshold lower cut = " << thrL\
              << " higher cut = " << thrH\
              << (is_step_mode ? "\nStep mode." : "")\
              << std::endl\
              << std::endl;

    auto T0 = bch::high_resolution_clock::now();

    // define filters (reusable)
    GaussianFilter<3> GF(gaussian_filter_size, gaussian_filter_sigma);
    FilterBase<3> SBX(3);
    SobelFilterX SX(2);
    SBX.mono2all((FilterBase<1>&)SX);
    FilterBase<3> SBY(3);
    SobelFilterY SY(2);
    SBY.mono2all((FilterBase<1>&)SY);

    // some image processors
    // for Canny algorithm we need at least 3 
    RGB_GZ_VHP<GPU<char>> *f_Ix_amp = new RGB_GZ_VHP<GPU<char>>( HEIGHT, WIDTH ) ; 
    RGB_GZ_VHP<GPU<char>> *f_Iy     = new RGB_GZ_VHP<GPU<char>>(*f_Ix_amp);
    RGB_GZ_VHP<GPU<char>> *f_dir    = new RGB_GZ_VHP<GPU<char>>(*f_Ix_amp);

    // Execution Policy for cuda kernel
    // threadIdx.z has to be 3 (or larger) 
    auto ExPola = ExecutionPolicy(dim3(16,16,3),dim3((HEIGHT+15)/16,(WIDTH+15)/16,1));

    std::cout << "    Processing " << FN << std::endl; 
    f_Ix_amp->load_from_file( FN ) ;
    if (f_Ix_amp->is_empty())
    {
        errorln("    File not found. Quit without processing." );
        delete f_Ix_amp; delete f_Iy; delete f_dir; 
        return;
    }

    // copy loaded image into f_Iy
    *f_Iy = *f_Ix_amp;

    f_Ix_amp->apply_filter_GPU<unsigned char>( GF,  ExPola );
    if (is_step_mode)
    {
        f_Ix_amp->to_bmp(FN + std::string(".gaussian.bmp")) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".R.gaussian.bmp"), 0) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".G.gaussian.bmp"), 1) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".B.gaussian.bmp"), 2) ;
    }

    f_Ix_amp->apply_filter_GPU<char>( SBX, ExPola );
    if (is_step_mode)
    {
        f_Ix_amp->to_bmp(FN + std::string(".sobel.x.bmp")) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".R.sobel.x.bmp"), 0) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".G.sobel.x.bmp"), 1) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".B.sobel.x.bmp"), 2) ;
    }

    // copy Ix into f_dir
    *f_dir = *f_Ix_amp;

    f_Iy->apply_filter_GPU<unsigned char>( GF,  ExPola );
    f_Iy->apply_filter_GPU<char>( SBY, ExPola );
    if (is_step_mode)
    {
        f_Iy->to_bmp(FN + std::string(".sobel.y.bmp")) ;
        f_Iy->to_bmp_one_channel(FN + std::string(".R.sobel.y.bmp"), 0) ;
        f_Iy->to_bmp_one_channel(FN + std::string(".G.sobel.y.bmp"), 1) ;
        f_Iy->to_bmp_one_channel(FN + std::string(".B.sobel.y.bmp"), 2) ;
    }

    f_Ix_amp->calculate_gradient_amplitude_GPU(*f_Iy, ExPola) ;
    if (is_step_mode)
    {
        f_Ix_amp->to_bmp(FN + std::string(".sobel.amp.bmp")) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".R.sobel.amp.bmp"), 0) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".G.sobel.amp.bmp"), 1) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".B.sobel.amp.bmp"), 2) ;
    }

    f_dir->calculate_gradient_direction_GPU(*f_Iy, ExPola) ;
    if (is_step_mode)
    {
        f_dir->to_bmp(FN + std::string(".sobel.dir.bmp")) ;
        f_dir->to_bmp_one_channel(FN + std::string(".R.sobel.dir.bmp"), 0) ;
        f_dir->to_bmp_one_channel(FN + std::string(".G.sobel.dir.bmp"), 1) ;
        f_dir->to_bmp_one_channel(FN + std::string(".B.sobel.dir.bmp"), 2) ;
    }

    f_Ix_amp->non_maximal_suppression_GPU(*f_dir, ExPola) ;
    if (is_step_mode)
    {
        f_Ix_amp->to_bmp(FN + std::string(".sobel.suppression.bmp")) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".R.sobel.suppression.bmp"), 0) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".G.sobel.suppression.bmp"), 1) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".B.sobel.suppression.bmp"), 2) ;
    }

    f_Ix_amp->rescaling_GPU(ExPola, cutoff) ;
    if (is_step_mode)
    {
        f_Ix_amp->to_bmp(FN + std::string(".rescale.bmp")) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".R.rescale.bmp"), 0) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".G.rescale.bmp"), 1) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".B.rescale.bmp"), 2) ;
    }

    f_Ix_amp->double_threshold_GPU(ExPola, thrL, thrH) ;
    if (is_step_mode)
    {
        f_Ix_amp->to_bmp(FN + std::string(".doublethreshold.bmp")) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".R.doublethreshold.bmp"), 0) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".G.doublethreshold.bmp"), 1) ;
        f_Ix_amp->to_bmp_one_channel(FN + std::string(".B.doublethreshold.bmp"), 2) ;
    }
    else
    {
        f_Ix_amp->to_bmp(FN + std::string(".canny.bmp")) ;
    }

    delete f_Ix_amp; delete f_Iy; delete f_dir; 

    auto Tf = timing(T0, "    Timing : total ") ;

    return ;
}


void print_help(char * program_name)
{
    std::cerr << "Usage: " << program_name << " <rgb.gz input file> [options]" << std::endl;
    std::cerr << "options: " << std::endl;
    std::cerr << "-t // in step mode" << std::endl;
    std::cerr << "-c <rescaling cutoff>" << std::endl;
    std::cerr << "-l <double threshold lower bound>" << std::endl;
    std::cerr << "-u <double threshold upper bound>" << std::endl;
    std::cerr << "-s <initial gaussian filter sigma>" << std::endl;
    std::cerr << "-n <initial gaussian filter size>" << std::endl;
}


std::tuple<std::string, int, double, unsigned int, unsigned int, unsigned int, bool> 
parseCommandLine(int argc, char *argv[])
{
    std::string FN(argv[1]);
    int gaussian_filter_size = 11; 
    double gaussian_filter_sigma = 5.0;
    unsigned int cutoff = 64u;
    unsigned int thrL = 8u;
    unsigned int thrH = 16u;
    bool is_step_mode = false;

    for (int i = 2; i < argc; i++)
    {
        std::string option(argv[i]);
        if (option.compare("-t") == 0)
        {
            is_step_mode = true;
            continue;
        }
        i++;
        std::string value(argv[i]);

        if (option.compare("-h") == 0)
        {
            print_help(argv[0]);
            exit(0);
        }
        else if (option.compare("-c") == 0)
        {
            // -c <rescaling cutoff>
            cutoff = (unsigned int)std::atoi(value.c_str());
            if (cutoff <= 0 || cutoff >=256)
            {
                errorln("Inappropriate rescaling cutoff. Exit.");
                exit(1);
            }
        }
        else if (option.compare("-l") == 0)
        {
            // -l <double threshold lower bound>
            thrL = (unsigned int)std::atoi(value.c_str());
            if (thrL <= 0 || thrL >=256)
            {
                errorln("Inappropriate double threshold lower bound. Exit.");
                exit(1);
            }
        }
        else if (option.compare("-u") == 0)
        {
            // -u <double threshold upper bound>
            thrH = (unsigned int)std::atoi(value.c_str());
            if (thrH <= 0 || thrH >=256)
            {
                errorln("Inappropriate double threshold upper bound. Exit.");
                exit(1);
            }
        }
        else if (option.compare("-s") == 0)
        {
            // -s <initial gaussian filter sigma>
            gaussian_filter_sigma = std::atof(value.c_str());
            if (gaussian_filter_sigma <= 0.0)
            {
                errorln("Inappropriate Gaussian filter sigma. Exit.");
                exit(1);
            }
        }
        else if (option.compare("-n") == 0)
        {
            // -s <initial gaussian filter size>
            gaussian_filter_size = std::atoi(value.c_str());
            if (gaussian_filter_size <= 3 || gaussian_filter_size > 200) 
            {
                errorln("Inappropriate Gaussian filter size. Exit.");
                exit(1);
            }
        }
        else 
        {
            errorln("Inappropriate options. Exit.");
            exit(1);
        }
    }

    if (thrH <= thrL)
    {
        errorln("Inappropriate double threshold upper/lower bound. Exit.");
        exit(1);
    }

    return {FN, gaussian_filter_size, gaussian_filter_sigma, cutoff, thrL, thrH, is_step_mode};
}


int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        print_help(argv[0]);
        exit(1) ;
    }
    auto[FN, gaussian_filter_size, gaussian_filter_sigma, cutoff, thrL, thrH, is_step_mode] =\
        parseCommandLine(argc, argv);
    
    CannyStepVHP(
        FN, 
        is_step_mode, 
        gaussian_filter_size, gaussian_filter_sigma, 
        cutoff, thrL, thrH
    );

    return 0;
}
