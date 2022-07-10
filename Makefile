DIR=./src

DIRH=./src/helper

COMPILER=nvcc

BOOST_FLAGS=-I/usr/lib -I/usr/include/boost -lboost_iostreams -lboost_system -lboost_thread -lboost_timer -lboost_chrono

CUDA_FLAGS=-I/usr/local/cuda/include -I/usr/local/cuda/lib64  -lcuda -lcudart 

.PHONY: clean build run

##################

build-q:
	$(COMPILER) --std c++17 -O3 ./src/deviceQuery.cu -o ./bin/deviceQuery.exe -I$(IDIR) -I$(DIRH) $(BOOST_FLAGS) $(CUDA_FLAGS)

run-q:
	./bin/deviceQuery.exe

q: build-q run-q


##################

clean:
	rm -rf  ./bin/*.exe  *.bmp   *.fltr   *.txt   /tmp/*.o   ./data/*.bmp   ./data/*.xy   ./data/*.txt

##################

build-test:
	$(COMPILER) --std c++17 -O3 -lineinfo ./src/test.cu -o ./bin/test.exe -I$(IDIR) $(BOOST_FLAGS) $(CUDA_FLAGS)

run-test:
	./bin/test.exe

test: clean build-test run-test

##################

build: build-canny build-test

run: run-test

all: clean build run

##################

build-canny: ./src/CannyStepVHP.cu
	$(COMPILER) --std c++17 -O3 -lineinfo ./src/CannyStepVHP.cu -o ./bin/CannyStepVHP.exe -I$(IDIR) $(BOOST_FLAGS) $(CUDA_FLAGS)
