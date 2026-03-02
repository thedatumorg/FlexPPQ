# FlexPPQ

# Requirements
- CMake 3.22.1 or newer
- g++ 11.4.0 or newer

# Pre-Build
1. Download [Eigen](https://gitlab.com/libeigen/eigen/-/releases/3.4.0) at external, unzip and rename it as eigen.
```
cd external
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip
mv eigen-3.4.0 eigen
```
2. Install Blas and Lapacke
```
sudo apt install -y libblas-dev liblapack-dev liblapacke-dev
sudo apt install -y libnuma-dev libarmadillo-dev numactl 
```

# Build
```
cd .. # Return back to the main dir
mkdir build && cd build
cmake ../
make -j8 # For parallel compiling
```

# Run
Please check run_demo.sh for detailed parameters introduction and settings.
```
cd .. # Return back to the main dir
bash run_demo.sh
```
