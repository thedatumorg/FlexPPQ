# FlexPPQ: Fast and Accurate Approximate Similarity Search via Flexible-bitwidth Partitioned Product Quantization

# Overview
Similarity search retrieves vectors similar to a query and underpins modern database systems. As datasets grow, approximate nearest neighbor (ANN) methods provide an efficient alternative to exact search by returning close, but not the true nearest neighbors. Among ANN methods, Product Quantization (PQ) splits vectors into subspaces encoded with learned codebooks, enabling memory-efficient and fast distance computation via lookup tables (LUTs) storing precomputed query-to-codeword distances. Hardware-accelerated PQ methods further improve throughput by quantizing LUTs into low-bit integers. However, prior work face four bottlenecks: (i) hardware-constrained fixed-bit per-subspace encoding (e.g., 4-bit), which sacrifices accuracy for efficiency; (ii) limited portability across CPU architectures (e.g., AVX-512 vs. AVX2); (iii) the difficulty of achieving high-fidelity representation with efficient indexing; and (iv) precision loss from coarse LUT quantization. To address these issues, we present FlexPPQ, a hardware-accelerated PQ method with three innovations:  (i) a flexible-bitwidth fast scan that supports varying numbers of subspaces and 4–10 bits per subspace, enabling—for the first time—true flexibility to match diverse hardware and performance requirements; (ii) a hierarchical partitioned PQ design that improves quantization fidelity via partition-aware codebooks while maintaining scalable performance by avoiding LUT construction for each partition; and (iii) thresholded LUT quantization that allocate more precision to smaller distances that is critical for top- k ranking. Our evaluation on 30 diverse datasets with rigorous statistical analysis shows that FlexPPQ achieves up to 6.3x and 7.1x speedups over QuickerADC and IVFPQFS, respectively—the strongest non-indexed and indexed hardware-accelerated PQ methods—while achieving higher recall with up to 65% lower bit budgets. FlexPPQ reconciles hardware efficiency with high precision quantization, eliminating the long-standing sacrifice of accuracy for efficiency in hardware-accelerated ANN search.

# Installment
## Requirements
- CMake 3.22.1 or newer
- g++ 11.4.0 or newer

## Pre-Build
1. Download [Eigen](https://gitlab.com/libeigen/eigen/-/releases/3.4.0) at external, unzip and rename it as eigen.
```
mkdir external
cd external
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip
mv eigen-3.4.0 eigen
```
2. Install Blas, Lapacke, amd necessary libs
```
sudo apt install -y libblas-dev liblapack-dev liblapacke-dev
sudo apt install -y libnuma-dev libarmadillo-dev numactl libtbb-dev
```

## Build
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
Then you can check results at metric/*.
