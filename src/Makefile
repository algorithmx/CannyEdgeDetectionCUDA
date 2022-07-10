DIR=./

DIRH=./helper

COMPILER=nvcc

BOOST_FLAGS=-I/usr/lib -I/usr/include/boost -lboost_iostreams -lboost_system -lboost_thread -lboost_timer -lboost_chrono

CUDA_FLAGS=-I/usr/local/cuda/include -I/usr/local/cuda/lib64  -lcuda -lcudart 

.PHONY: clean build run

##################

build-q: deviceQuery.cu
	clear
	$(COMPILER) --std c++17 -O3 deviceQuery.cu -o deviceQuery.exe -I$(IDIR) -I$(DIRH) $(BOOST_FLAGS) $(CUDA_FLAGS)

run-q:
	./deviceQuery.exe

q: build-q run-q


##################

clean:
	rm -rf  *.exe   *.bmp   *.fltr   *.txt   /tmp/*.o   ../data/*.bmp   ../data/*.xy   ../data/*.txt

##################

build-test: test.cu
	clear
	$(COMPILER) --std c++17 -O3 -lineinfo test.cu -o test.exe -I$(IDIR) $(BOOST_FLAGS) $(CUDA_FLAGS)

run-test:
	./test.exe $(ARGS)

test: clean build-test run-test

##################

build: build-canny build-test

run: run-test

all: clean build run

##################

build-canny: CannyStepVHP.cu
	clear
	$(COMPILER) --std c++17 -O3 -lineinfo CannyStepVHP.cu -o CannyStepVHP.exe -I$(IDIR) $(BOOST_FLAGS) $(CUDA_FLAGS)
