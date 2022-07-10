# CannyEdgeDetectionCUDA

This is the course project for [CUDA at Scale For The Enterprise](https://www.coursera.org/learn/cuda-at-scale-for-the-enterprise) on Coursera. 

## Overview

In this project, I have implemented the [Canny edge detection algorithm](https://en.wikipedia.org/wiki/Canny_edge_detector) using C++ and CUDA. The major references are:

+ []() ([backup copy](./references/Week4.pdf))
+ []() ([backup copy](./references/))


The compilation environment is 

```
(base) dabajabaza@XXXX:~/jianguoyun/Nutstore/CUDA_courses/course3_project$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```


## Setup

To set up for a test and trial run, first unpack the project into a folder. The source code is in `./src` and the compiled binaries will be put in the folder `./bin`. The  `Makefile`  summarizes the compilation work. All you have to do is to enter the project folder in the command line prompt and run the command `make all` :

```
(base) dabajabaza@XXXX:~/jianguoyun/Nutstore/CUDA_courses/course3_project$ make all
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
(base) dabajabaza@XXXX:~/jianguoyun/Nutstore/CUDA_courses/course3_project$ ./bin/test.exe 
*** test_Mem() started.
...
...
```
because it assumes `./data` to be the location of the data folder. This does not affect the other binary `CannyStepVHP.exe` . 


If you want to skip the test, just run `make clean build-canny` instead. It will generate the binary  `CannyStepVHP.exe`  in the  `./bin`  folder. 

## Important details of the code

### `CannyStepVHP()` function 

### `class Mem`

### `class ImageBase`


## The Visible Human Project

The test data is obtained from the Visible Human Project website. 

## Sample output


## Proof of run


