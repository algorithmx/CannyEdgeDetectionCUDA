# CannyEdgeDetectionCUDA

This is the course project for [CUDA at Scale For The Enterprise](https://www.coursera.org/learn/cuda-at-scale-for-the-enterprise) on Coursera. 

## Overview

In this project, I have implemented the [Canny edge detection algorithm](https://en.wikipedia.org/wiki/Canny_edge_detector) using C++ and CUDA. The major references are:

+ [Week 4: Image Filtering and Edge Detection](https://sbme-tutorials.github.io/2018/cv/notes/4_week4.html) ([backup copy](./references/Week4.pdf))
+ [CANNY EDGE DETECTION (by Justin Liang)](https://justin-liang.com/tutorials/canny/) ([backup copy](./references/Canny_Liang.pdf))
+ [CAP4453-Robot Vision: Lecture 6-Canny Edge Detection (by Dr. Ulas Bagci)](http://www.cs.ucf.edu/~bagci/teaching/robotvision18/Lec6.pdf) ([backup copy](./references/CAP4453_Lecture6.pdf))


---

## Compilation environment

```
(base) XXX@XXX:~/PROJECT_ROOT_FOLDER$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```

Major compiler options:

+ C++
```
 --std c++17 -O3 
```
+ CUDA
```
 -lcuda -lcudart 
```
+ Boost
```
 -lboost_iostreams -lboost_system -lboost_thread -lboost_timer -lboost_chrono
```

---

## Setup

To set up for a test and trial run, first unpack the project into a folder. The source code is in `./src` and the compiled binaries will be put in the folder `./bin`. The  `Makefile`  summarizes the compilation work. All you have to do is to enter the project folder in the command line prompt and run the command `make all` :

```
(base) XXX@XXX:~/PROJECT_ROOT_FOLDER$ make all
rm -rf  ./bin/*.exe  *.bmp   *.fltr   *.txt   /tmp/*.o   ./data/*.bmp   ./data/*.xy   ./data/*.txt
nvcc --std c++17 -O3 -lineinfo ./src/CannyStepVHP.cu -o ./bin/CannyStepVHP.exe -I -I/usr/lib -I/usr/include/boost -lboost_iostreams -lboost_system -lboost_thread -lboost_timer -lboost_chrono -I/usr/local/cuda/include -I/usr/local/cuda/lib64  -lcuda -lcudart 
nvcc --std c++17 -O3 -lineinfo ./src/test.cu -o ./bin/test.exe -I -I/usr/lib -I/usr/include/boost -lboost_iostreams -lboost_system -lboost_thread -lboost_timer -lboost_chrono -I/usr/local/cuda/include -I/usr/local/cuda/lib64  -lcuda -lcudart 
./bin/test.exe
*** test_Mem() started.
...
...
```

Note that you must call `test.exe` from the root of project folder, as 

```
(base) XXX@XXX:~/PROJECT_ROOT_FOLDER$ ./bin/test.exe 
*** test_Mem() started.
...
...
```
because it assumes `./data` to be the location of the data folder. This does not affect the other binary `CannyStepVHP.exe` . 


If you want to skip the test, just run `make clean build-canny` instead. It will generate the binary  `CannyStepVHP.exe`  in the  `./bin`  folder. 

In case that the link to the Boost library fails, check the `BOOST_FLAGS` in `Makefile`  in particular the path follows the `-I` option to include appropriate paths for the Boost library in your OS.  

---

## Important details of the code

### `CannyStepVHP()` function 

This function summarizes the major steps of the Canny edge detection algorithm:
+ Apply the Gaussian filter  
  Line 59, 78 in `src/CannyStepVHP.cu` :  `apply_filter_GPU( GF ...)`
+ Apply the Sobel filters  
  Line 67 in `src/CannyStepVHP.cu` :  `apply_filter_GPU( SBX ...)`
  Line 79 in `src/CannyStepVHP.cu` :  `apply_filter_GPU( SBY ...)`
+ Compute the amplitude of gradient as $\sqrt{I_x^2+I_y^2}$, where $I_x$ and $I_y$ are the results after the Sobel filter.
  Line 87 in `src/CannyStepVHP.cu` :  `calculate_gradient_amplitude_GPU( ...)`
+ Compute the direction of gradient as $\arctan(I_y/I_x)$, where $I_x$ and $I_y$ are the results after the Sobel filter.
  Line 95 in `src/CannyStepVHP.cu` :  `calculate_gradient_direction_GPU( ...)`
+ Suppress the non maximal pixels in the amplitude of gradient.
  Line 103 in `src/CannyStepVHP.cu` :  `non_maximal_suppression_GPU( ...)`
+ Rescale the gradient amplitude to enhance the contrast in the regions of interest.
  Line 111 in `src/CannyStepVHP.cu` :  `rescaling_GPU( ...)`
+ Hysteresis edge tracking.
  Line 119 in `src/CannyStepVHP.cu` :  `double_threshold_GPU( ...)`


### `class Mem`

The file `Mem.hpp` is a primitive implementation of memory manager for the image processing tasks. Each `Mem<T>` object contains a pointer to some dynamically allocated memory, either on host system RAM (`CPU<T>`, `CPUpinned<T>`) or on device GPU RAM (`GPU<T>`, `GPUpinned<T>`, `GPUstream<T>`). Theses classes hide the details of memory allocation and synchronization.  


### `class ImageBase<NCH,M>`

This is a class abstracted from the task of processing RGB interleaving images in the Visible Human Project. The template parameter `NCH` is the number of channels in the image. For RGB images it is 3, whereas for the CT scan images, it is 2. For monochromatic images (such as filters), this number is 1. The other template parameter `M` is the memory manager discussed earlier. They can be `CPU<char>`, `CPUpinned<char>`, `GPU<char>`, `GPUpinned<char>` or `GPUstream<char>`.

Special care has been taken to interprete the bytes for the image pixel data. Most of the time the bytes are interpreted as `unsigned char`. When the Sobel filter is involved, the bytes in some of the intermediate images are interpreted as `char` with sign. Without careful treatment of the raw bytes, the Canny edge detection algorithm will not work. 

The class design allows for customized codes for different purposes. 

---

## The Visible Human Project

The test data is obtained from the [Visible Human Project (VHP)](https://en.wikipedia.org/wiki/Visible_Human_Project) website:

https://www.nlm.nih.gov/research/visible/visible_human.html

Despite controversy, this heroic project is a great contribution to the knowledge in Human Anatomy, enables the construction of realistic 3D models of human organs for physiological simulations. 

To construct a 3D model of human body, one should start with the images of the cross sectional images in the [VHP database](https://www.nlm.nih.gov/databases/download/vhp.html). A good idea is to extract color edges in these images, then align them vertically for 3D mesh generation. Here is a typical RGB image:



The goal of my project is to extract the color edges in the RGB images in the VHP database. I have only implemented the interface for images in the Male body folder.  

---

## Sample output

If the compilation is successful, the following commands 

```
(base) XXX@XXX:~/PROJECT_ROOT_FOLDER$ cd data

(base) XXX@XXX:~/PROJECT_ROOT_FOLDER$ wget https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Male-Images/70mm/fullbody/1715.rgb.gz

(base) XXX@XXX:~/PROJECT_ROOT_FOLDER$ cd ..

(base) XXX@XXX:~/PROJECT_ROOT_FOLDER$ ./bin/CannyStepVHP.exe ./data/1715.rgb.gz -n 17 -s 9.0 -l 24 -u 48 -c 96

CannyStepVHP() started with file ./data/1715.rgb.gz
    Gaussian filter size = 17 , sigma = 9
    Rescaling cutoff = 96
    Double threshold lower cut = 24 higher cut = 48

    Processing ./data/1715.rgb.gz
    Timing : RGB_VHP::_load()  : 321 milliseconds
    Timing : total  : 1626 milliseconds
```

will generate the following images:

![Male-fullbody-1715](./sample/1715.rgb.gz.bmp)

![Male-fullbody-1715-edge](./sample/1715.rgb.gz.canny.bmp)

They are stored in the `./sample` folder.

---

Here is another example.

```
(base) XXX@XXX:~/PROJECT_ROOT_FOLDER$ cd data

(base) XXX@XXX:~/PROJECT_ROOT_FOLDER$ wget https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Male-Images/70mm/fullbody/1092.rgb.gz

(base) XXX@XXX:~/PROJECT_ROOT_FOLDER$ cd ..

(base) dabajabaza@XXXX:~/jianguoyun/Nutstore/CUDA_courses/course3_project$ ./bin/CannyStepVHP.exe ./data/1092.rgb.gz -n 17 -s 11.0 -l 4 -u 24 -c 64
```

![Male-fullbody-1092](./sample/1092.rgb.gz.bmp)

![Male-fullbody-1092-edge](./sample/1092.rgb.gz.canny.bmp)

---

## Proof of run


