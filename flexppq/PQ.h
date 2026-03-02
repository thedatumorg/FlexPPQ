#ifndef PQ_H
#define PQ_H

#include <iostream>
#include <sstream>
#include <type_traits>
#include <Eigen/Eigenvalues>
#include <chrono>
#include <numeric>
#include "glpk.h"

#include "KMeans.hpp"

#include "utils/Types.hpp"
#ifdef __AVX2__
#include "utils/AVXUtils.hpp"
#endif
#include "utils/IO.hpp"
#include "utils/Math.hpp"
#include "utils/Experiment.hpp"
#include "utils/Heap.hpp"
#include "utils/DataQuantization/DataQuantizer.h"

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

extern "C" {
#ifndef FINTEGER
#define FINTEGER int
#endif

int sgemm_ (
        const char *transa, const char *transb, 
        FINTEGER *m, FINTEGER *n, FINTEGER *k, 
        const float *alpha, const float *a,
        FINTEGER *lda, const float *b,
        FINTEGER *ldb, float *beta,
        float *c, FINTEGER *ldc);
}


struct BucketTopK_u8 {
    int k;
    int w;              
    uint32_t kept;        
    std::array<uint32_t, 256> hist{};
    std::vector<int> ids_flat;          
    std::array<uint16_t, 256> sz{};   

    explicit BucketTopK_u8(int k_) : k(k_), w(255), kept(0), ids_flat(256 * k_) {
        hist.fill(0);
        sz.fill(0);
    }

    inline void clear_bucket(int d) { sz[d] = 0; hist[d] = 0; }

    inline void push_id(int d, int id) {
        uint16_t &s = sz[d];
        if (s < (uint16_t)k) {
            ids_flat[d * k + s] = id;
            ++s;
        }
    }

    inline void shrink_to_k() {
        while (w > 0) {
            uint32_t hw = hist[w];
            if (kept - hw < (uint32_t)k) break;
            kept -= hw;
            clear_bucket(w);
            --w;
        }

        uint32_t below = kept - hist[w];
        uint32_t need_at_w = (uint32_t)k - below;   
        if (need_at_w < sz[w]) sz[w] = (uint16_t)need_at_w;

        kept = (uint32_t)k;
    }

    inline void observe(uint8_t dist, int id) {
        int d = (int)dist;
        if (d > w) return;       
        hist[d] += 1;
        kept += 1;
        push_id(d, id);

        if (kept > (uint32_t)k) shrink_to_k();
    }

    template <class TopK>
    inline void finalize_to(TopK& res) const {
        const uint8_t INF = std::numeric_limits<uint8_t>::max();

        int out = 0;
        for (int d = 0; d <= w && out < k; ++d) {
            uint16_t s = sz[d];
            for (uint16_t j = 0; j < s && out < k; ++j) {
                res.dist[out] = (uint8_t)d;
                res.ids[out]  = ids_flat[d * k + j];
                ++out;
            }
        }
        for (; out < k; ++out) {
            res.dist[out] = INF;
            res.ids[out]  = -1;
        }
    }
};



class PQ {
public:

  using CodewordType = uint16_t;

  int mSubspaceNum;
  int mSubsLen;
  int mCentroidsNum;
  thread_local static LUTType lut;
  double lutSec = 0;
  double kernelSec = 0;
  double keyKernalSec = 0;
  size_t pruntCount = 0;
  std::vector<int> perm_sub;
  std::vector<int> inv_perm_sub_;
  std::vector<long long> pruneMarks;
  // int *toOriginalID;
  int idBias;

  CentroidsPerSubsType mCentroidsPerSubs;
  CentroidsPerSubsTypeColMajor mCentroidsPerSubsCMajor;
  
  int mXTrainRows, mXTrainCols;
  // CodebookType mCodebook;
  RowMatrix<CodewordType> mCodebook;
  ColMatrix<uint8_t> mSmallCodebook;
  int codebookRow;
  // CodebookTypeColMajor<> mCodebookCMajor;
  // CodebookTypeColMajor<uint16_t> mCodebookCMajor16;

  void normalize(RowMatrixXf &data);
  void train(RowMatrixXf &XTrain, bool verbose);

  struct IVFList {
    // local_id -> original_id
    std::vector<int> idmap;

    // PQ codes for vectors in this list:
    // col-major matrix: rows = R(list size), cols = M(subspaces)
    ColMatrix<uint8_t> codes;
    std::vector<uint64_t> codeHiPacked; // Only use when numCentroids >= 512
    std::vector<uint64_t> codeHiPacked9;
    ColMatrix<uint16_t> tailCodes;
    int tailStart = 0;   
    int tailLen   = 0;    

    void save(std::ofstream& out) const{
      saveIterable(idmap, idmap.size(), out);
      saveMatrix(codes, out);
      saveIterable(codeHiPacked, codeHiPacked.size(), out);
      saveIterable(codeHiPacked, codeHiPacked9.size(), out);
      saveMatrix(tailCodes, out);
      saveOneData(tailStart, out);
      saveOneData(tailLen, out);
    }

    void load(std::ifstream& in) {
      loadIterable(idmap, in);
      codes = loadMatrix<ColMatrix<uint8_t>>(in);
      loadIterable(codeHiPacked, in);
      loadIterable(codeHiPacked9, in);
      tailCodes = loadMatrix<ColMatrix<uint16_t>>(in);
      loadOneData(tailStart, in);
      loadOneData(tailLen, in);
    }
  };

  struct IVFIndex {
    int nlist = 0;
    int mIVFLists = 256;
    RowMatrixXf coarseCentroids;     // (nlist x dim)
    Eigen::VectorXf coarseNorm2;
    std::vector<IVFList> lists;      // size = nlist
    void save(std::ofstream& out) const{
      saveOneData(nlist, out);
      saveOneData(mIVFLists, out);
      saveMatrix(coarseCentroids, out);
      saveEigenVector(coarseNorm2, out);
      assert(nlist == lists.size());
      for(int i = 0; i < nlist; ++i) {
        lists[i].save(out);
      }
    }
    void load(std::ifstream& in) {
      loadOneData(nlist, in);
      loadOneData(mIVFLists, in);
      coarseCentroids = loadMatrix<RowMatrixXf>(in);
      loadEigenVector(coarseNorm2, in);
      lists.resize(nlist);
      for(int i = 0; i < nlist; ++i) {
        lists[i].load(in);
      }
    }
  };

                // nlist
  IVFIndex mIVF;

  // global PQ codebook (already in your code)
  //std::vector<RowMatrixXf> mCentroidsPerSubs; // [M], each (K x dsub)
  void trainIVF(RowMatrixXf &XTrain, const int nlist, bool verbose);
  void encodeIVF(const RowMatrixXf &XTrain, bool verbose=false);

  struct ID {
    uint8_t clusterID;
    int internalID;
  };
  
  struct TopKHeap {
      static constexpr int bufferK = 400;
      uint8_t dist[bufferK];

      ID ids[bufferK];
      int realK;

      TopKHeap() {
      }

      void reset(const int realK) {
        this->realK = realK;
        const uint8_t INF = std::numeric_limits<uint8_t>::max();
        // const uint8_t INF = 200;
        std::fill_n(dist, realK, INF);
        std::fill_n(ids, realK, ID{0, 0});
      }

      inline uint8_t worst() const { return dist[0]; }

      inline void siftDown(int i) {
          while (true) {
              int c0 = i * 4 + 1;
              if (c0 >= realK) break;
              int best = c0;
              int c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
              if (c1 < realK && dist[c1] > dist[best]) best = c1;
              if (c2 < realK && dist[c2] > dist[best]) best = c2;
              if (c3 < realK && dist[c3] > dist[best]) best = c3;

              if (dist[i] >= dist[best]) break;
              std::swap(dist[i], dist[best]);
              std::swap(ids[i],  ids[best]);
              i = best;
          }
      }

      inline void push(uint8_t d, const ID &id) {
          // if (d > dist[0]) return;
          dist[0] = d;
          ids[0]  = id;
          siftDown(0);
      }

    };


  struct SubspaceScore {
    int s;
    double topCMass;   // sum of topC frequencies
    double maxFreq;    // maximum frequency
  };

  float subspaceDistSq(
    const RowMatrixXf& XTrain,
    int rowIdx,
    int sIdx,
    int subsLen,
    const Eigen::MatrixXf& centroids,  // [Ks, subsLen]
    int code
  ); 
  void encode(const RowMatrixXf &XTrain, std::vector<int> &, const float alpha);
  void encode(const RowMatrixXf &XTrain);
  template<class T>
  void encodeImpl(const RowMatrixXf &XTrain, T &codebook);
  template<class T>
  void encodeImplReorder(
    const RowMatrixXf& XTrain,
    std::vector<int>&, 
    T& codebook,
    int topC,
    float alpha,
    int codeThreshold
  );

  struct Group{
    ColMatrix<uint8_t> mSmallCodebook;
    std::vector<int> pqID;
    bool isLow[8];
  };
  std::array<Group, 257> groups;
  void buildGroups();



  RowMatrixXf decode();

  LabelDistVecF search(const RowMatrixXf &XTest, const int k, bool verbose=false);
  template<typename TargetDType=float>
  LabelDistVecF searchOne(const RowVectorXf &XTest, const int k, bool simd, bool verbose,
    TopKHeap &topKHeap, int idBias, bool retrainQuant);

  void searchOneIVF(const RowVectorXf &XTest, const int k, int nprobe, bool simd, bool verbose,
    TopKHeap &topKHeap, int idBias, DataQuantizer &quantizer);

  LabelDistVecF refine(const RowMatrixXf &XTest, const LabelDistVecF &answersIn, const RowMatrixXf &XTrain, const int k);

  void prepareSmallCodebook();

  void parseMethodString(std::string methodString);

  void loadMetaData(const std::string& filepath);
  void saveMetaData(const std::string& filepath);

  void CreateLUT(const RowVectorXf &query) {
    lut = LUTType(mCentroidsNum, mSubspaceNum);
    lut.setZero();
    float * lutPtr = lut.data();
    for (int subs=0; subs<mSubspaceNum; subs++) {
      const float* qsub = query.data() + subs * mSubsLen;
      fvec_L2sqr_ny(lutPtr, qsub, mCentroidsPerSubs[subs].data(), mSubsLen, mCentroidsNum);
      lutPtr += lut.rows();
    }
  }


  struct SubspaceCentroidStats {
    // [subspace][centroid] -> count
    std::vector<std::vector<size_t>> counts;

    // [subspace][centroid] -> ratio in [0,1]
    std::vector<std::vector<double>> ratios;

    std::vector<size_t> batchCounts;
    std::vector<double> batchRatios;
  };

  SubspaceCentroidStats
  computeCentroidDistribution() const{
      const size_t numVectors = mSmallCodebook.rows();

      printf("#Subspaces=%d, #Centroids=%d, #Vectors=%d\n", int(mSubspaceNum), int(mCentroidsNum), int(numVectors));


      SubspaceCentroidStats stats;
      stats.counts.assign(mSubspaceNum,
                          std::vector<size_t>(mCentroidsNum, 0));
      stats.ratios.assign(mSubspaceNum,
                          std::vector<double>(mCentroidsNum, 0.0));

      // --------------------------------------------------
      // Count
      // --------------------------------------------------
      for (size_t i = 0; i < numVectors; ++i) {
          for (size_t s = 0; s < mSubspaceNum; ++s) {
              uint8_t centroid = mSmallCodebook(i, s);
              stats.counts[s][centroid]++;
          }
      }

      // --------------------------------------------------
      // Normalize to ratio
      // --------------------------------------------------
      for (size_t s = 0; s < mSubspaceNum; ++s) {
          for (size_t c = 0; c < mCentroidsNum; ++c) {
              stats.ratios[s][c] =
                  static_cast<double>(stats.counts[s][c]) /
                  static_cast<double>(numVectors);
          }
      }

      constexpr int stepSize = 32;
      stats.batchRatios.resize(mSubspaceNum);
      stats.batchCounts.resize(mSubspaceNum);
      int batchNum = numVectors / stepSize;
      for (size_t i = 0; i < batchNum * stepSize; i += stepSize) {
          for (size_t s = 0; s < mSubspaceNum; ++s) {
              bool allSmaller = true;
              for(int j = 0; j < stepSize;++j) {
                uint8_t centroid = mSmallCodebook(i+j, s);
                if(centroid >= 64) {
                  allSmaller = false;
                  break;
                }
              }
              if(allSmaller) {
                stats.batchCounts[s]++;
              }
          }
      }
      for (size_t s = 0; s < mSubspaceNum; ++s) {
          stats.batchRatios[s] =
              static_cast<double>(stats.batchCounts[s]) /
              static_cast<double>(batchNum);
      }

      return stats;
  }

  LabelDistVecF PQScanSIMD_float_LUT(const RowVectorXf &XTest, const std::vector<int>&ids);
  template<int nSubspaces>
  LabelDistVecF computeFloatDistImpl(const RowVectorXf &XTest, const std::vector<int>&ids);

  LabelDistVecF computeFloatDist(const RowVectorXf &XTest, const std::vector<int>&ids);

  struct Distribution{
    std::vector<std::vector<double>> dists; // dists[queryIdx][Distances]
    std::vector<std::vector<uint8_t>> quantizedDists;
    double min, top100min, top200min, top1000min;
    double all1percent, all5percent, all10percent, mean, median, max;
    double qmax, qmin, lutMean, lutMedian;
  };
  Distribution getFloatDistDistribution(const RowMatrixXf &XTest);

  struct SingleDistribution {
    std::vector<double> dists;
    std::vector<uint8_t> quantizedDists;
  };
  SingleDistribution getSingleDistribution(const RowVectorXf &XTest);

private:
  template<typename TargetDType=float>
  void searchHeap(ColMatrix<TargetDType> &lut, const int k, int q_idx, TopKHeap &res);
  template<typename TargetDType=uint8_t, int numCentroids=32, int numSubspaceNum>
  void searchHeapSIMD(ColMatrix<TargetDType> &lut, const int k, int q_idx, TopKHeap &res);
  template<typename TargetDType=uint8_t, int numCentroids=32, int numSubspaceNum>
  void searchHeapSIMD(ColMatrix<TargetDType> &lut, const int k, int q_idx, TopKHeap &res
    , ColMatrix<uint8_t> &curCodes, std::vector<int>& idmap);
  template<typename TargetDType=uint8_t, int numCentroids=64, int numSubspaceNum>
  void searchHeapSIMDLargeCentroids(ColMatrix<TargetDType> &lut, const int k, int q_idx, TopKHeap &res);
  template<typename TargetDType=uint8_t, int numCentroids=64, int numSubspaceNum>
  void searchHeapSIMDLargeCentroids(ColMatrix<TargetDType> &lut, const int k, int q_idx, TopKHeap &res
    , ColMatrix<uint8_t> &curCodes, std::vector<int>& idmap);
  template<typename TargetDType=uint8_t, int numSubspaceNum>
  void searchHeapSIMD128Centroids(
    ColMatrix<TargetDType> &lut, const int k, int q_idx, 
    TopKHeap &res);
  template <typename TargetDType, int numSubspaceNum>
  void searchHeapSIMD128Centroids(
    ColMatrix<TargetDType>& lut, const int k, int q_idx,
    TopKHeap& res, ColMatrix<uint8_t> &curCodes, std::vector<int>& idmap);
  // template<int numSubspaceNum>
  // void searchHeapSIMDFloat(ColMatrix<float> &lut, const int k, int q_idx, TopKHeap &res);

  
  template<int numSubspaceNum>
  void searchHeapSIMD256Centroids(ColMatrix<uint8_t> &lut, const int k, 
    TopKHeap &res, ColMatrix<uint8_t>& curCodes, std::vector<int>& idmap);
  template<int numSubspaceNum>
  void searchHeapSIMD512Centroids(ColMatrix<uint8_t> &lut, const int k, 
    TopKHeap &res, ColMatrix<uint8_t>& curCodes, ColMatrix<uint16_t>& tailCodes, const uint64_t* codeHiPacked, std::vector<int>& idmap);
  template<int numSubspaceNum>
  void searchHeapSIMD1024Centroids(ColMatrix<uint8_t> &lut, const int k, 
    TopKHeap &res, ColMatrix<uint8_t>& curCodes, ColMatrix<uint16_t>& tailCodes, 
    const uint64_t* codeHiPacked, const uint64_t* codeHiPacked9, std::vector<int>& idmap);

};

template<typename TargetDType>
inline TargetDType unsigendSaturatedAdd(TargetDType a, TargetDType b) {
  TargetDType sum = a + b;
  if ((sum < a) || (sum < b))
      return ~((TargetDType)0);
  else
      return sum;
}

// this method is only for simulating the result, not for improving efficiency
template<typename TargetDType>
void PQ::searchHeap(ColMatrix<TargetDType> &lut, const int k, int q_idx, TopKHeap &res) {

  // codebook is row-major
  CodewordType* codes = mCodebook.data();

  if constexpr (std::is_same<TargetDType, float>::value) {
    if(mSubspaceNum % 8 == 0) {
      for (int i = 0; i < mCodebook.rows(); i++) {
        TargetDType dist = 0;
    
        // lut is column major, so luts+=ksub is to move to the next subspace' distance table
        const TargetDType * luts = lut.data();
    
        const int ksub = lut.rows();

        for (int col=0; col < mSubspaceNum; col += 8) {
          float dism = 0;
          dism  = luts[*codes++]; luts += ksub;
          dism += luts[*codes++]; luts += ksub;
          dism += luts[*codes++]; luts += ksub;
          dism += luts[*codes++]; luts += ksub;
          dism  = luts[*codes++]; luts += ksub;
          dism += luts[*codes++]; luts += ksub;
          dism += luts[*codes++]; luts += ksub;
          dism += luts[*codes++]; luts += ksub;
          dist += dism;
        }
    
        
        if (dist < res.worst()) {        
          res.push(dist, ID{idBias,  i});         
        }
      }
    }
    else if(mSubspaceNum % 4 == 0) {
      for (int i = 0; i < mCodebook.rows(); i++) {
        TargetDType dist = 0;
    
        // lut is column major, so luts+=ksub is to move to the next subspace' distance table
        const TargetDType * luts = lut.data();
    
        const int ksub = lut.rows();

        for (int col=0; col < mSubspaceNum; col += 4) {
          float dism = 0;
          dism  = luts[*codes++]; luts += ksub;
          dism += luts[*codes++]; luts += ksub;
          dism += luts[*codes++]; luts += ksub;
          dism += luts[*codes++]; luts += ksub;
          dist += dism;
        }
    
        
        if (dist < res.worst()) {        
          res.push(dist, ID{idBias,  i});         
        }
    
      }
    }
  } else {
    for (int i = 0; i < mCodebook.rows(); i++) {
      TargetDType dist = 0;
  
      // lut is column major, so luts+=ksub is to move to the next subspace' distance table
      const TargetDType * luts = lut.data();
  
      const int ksub = lut.rows();
  
      for (int col=0; col < mSubspaceNum; col++) {
        if constexpr (!std::is_same<TargetDType, float>::value){
          dist = unsigendSaturatedAdd<TargetDType>(dist, luts[*codes++]);
        } else {
          dist += luts[*codes++];
        }
        luts += ksub;
      }
      
      if (dist < res.worst()) {        
        res.push(dist, ID{idBias,  i});         
      }
    }
  }

 

  //f::heap_reorder<f::CMax<TargetDType, int>> (k, heap_dis, heap_ids);
}
#include <iomanip>
inline void print_m256i_uint8(__m256i data) {
    alignas(32) uint8_t buffer[32]; 
    _mm256_store_si256(reinterpret_cast<__m256i*>(buffer), data);

    for (int i = 0; i < 32; ++i) {
        std::cout << std::setw(3) << static_cast<int>(buffer[i]) << " ";
    }
    std::cout << std::endl;
}

inline void print_m256i_uint8(__m128i data) {
    alignas(32) uint8_t buffer[16]; 
    _mm_store_si128(reinterpret_cast<__m128i*>(buffer), data);

    for (int i = 0; i < 16; ++i) {
        std::cout << std::setw(3) << static_cast<int>(buffer[i]) << " ";
    }
    std::cout << std::endl;
}

inline void print_m256i_uint16(__m256i data) {
    alignas(32) uint16_t buffer[16]; 
    _mm256_store_si256(reinterpret_cast<__m256i*>(buffer), data);

    for (int i = 0; i < 16; ++i) {
        std::cout << std::setw(5) << static_cast<int>(buffer[i]) << " ";
    }
    std::cout << std::endl;
}

inline void print_m256i_uint32(__m256i data) {
    alignas(32) uint32_t buffer[8]; 
    _mm256_store_si256(reinterpret_cast<__m256i*>(buffer), data);

    for (int i = 0; i < 8; ++i) {
        std::cout << std::setw(10) << static_cast<int>(buffer[i]) << " ";
    }
    std::cout << std::endl;
}

inline void print_m256_float(__m256 data) {
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, data);

  for (int j = 0; j < 8; ++j) {
      std::cout << tmp[j];
      if (j < 7) std::cout << ", ";
  }
  std::cout << std::endl;

}

inline void print_vector(std::vector<int> data, const bool b1=true) {
    for (int i = 0; i < data.size(); ++i) {
        if(b1)
          std::cout << std::setw(3) << static_cast<int>(data[i]) << " ";
        else
          std::cout << std::setw(5) << static_cast<int>(data[i]) << " ";
    }
    std::cout << std::endl;
}


// // TODO load heap max into the register
// template <int numSubspaceNum>
// void PQ::searchHeapSIMDFloat(ColMatrix<float>& lut, const int k, int q_idx,
//                         TopKHeap& res) {


//     const int ksub = lut.rows();       // 子空间距离表的行数

//     // Step 1: 预加载所有子空间的 LUT 数据到寄存器
//     __m256 lut_registers[16]; // 最多支持 16 个子空间
//     for (int col = 0; col < numSubspaceNum; ++col) {
//         // 直接加载 256 位的 8 个 float 数据
//         lut_registers[col] = _mm256_loadu_ps(reinterpret_cast<const float*>(lut.data() + col * ksub));
//     }

//     // mSmallCodebook is col-major
//     uint8_t* codes = mSmallCodebook.data();

//     // Step 2: 遍历 Codebook 进行搜索
//     int stepSize;
//     const int codebookRow = mSmallCodebook.rows();

//     __m256i heap_top;
//     // whether there is a new dist less than heap_top
//     int cmp_mask = 0;
//     stepSize = 8;
//     heap_top = _mm256_set1_ps(std::numeric_limits<float>::max());


//     for (int i = 0; i < codebookRow; i += stepSize) {

//         __m256 acc = _mm256_setzero_ps(); // 初始化累积距离寄存器

//         for (int col = 0; col < numSubspaceNum; ++col) {
//           __m128i indices_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + codebookRow*col));
//           __m256i indices_32 = _mm256_cvtepu8_epi32(indices_8);
//           __m256 dist_vector = _mm256_permutevar8x32_ps(lut_registers[col], indices_32);
//           acc = _mm256_add_ps(acc, dist_vector);
//         }

//         codes += stepSize;
//         __m256 cmp_result;
//         cmp_result = _mm256_cmp_ps(acc, heap_top, _CMP_LT_OQ);
//         cmp_mask = _mm256_movemask_ps(cmp_result);

//         // Step 4: 比较并更新堆
//         if(cmp_mask){
//           alignas(32) float temp[8];
//           _mm256_store_ps(temp, acc);
//           for (int j = 0; j < stepSize; ++j) {
//               uint8_t dist = temp[j];
//               if (dist < res.worst()) { 
//                   res.push(dist, ID{idBias,  i + j});
//               }
//           }

//           heap_top = _mm256_set1_epi8(res.worst());
          
//         }
//     }

//     //f::heap_reorder<f::CMax<float, int>>(k, heap_dis, heap_ids);
// }

// shuffle: https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/mm256-shuffle-epi8.html
// a shuffle can look up 32 indices once, each index must be ranged [0x00~0x0F]
template <typename TargetDType, int numCentroid, int numSubspaceNum>
void PQ::searchHeapSIMD(ColMatrix<TargetDType>& lut, const int k, int q_idx,
                        TopKHeap& res) {

    const int ksub = lut.rows();     

    __m256i lut_registers[numSubspaceNum];
    if constexpr (std::is_same<TargetDType, uint8_t>::value && numCentroid==16){
      for (int col = 0; col < numSubspaceNum; ++col) {
          __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut.data() + col * ksub));
          lut_registers[col] = _mm256_set_m128i(lut128, lut128);
      }
    } else {
      for (int col = 0; col < numSubspaceNum; ++col) {
          lut_registers[col] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + col * ksub));
      }
    }

    // mSmallCodebook is col-major
    uint8_t* codes = mSmallCodebook.data();

    int stepSize;
    const int codebookRow = mSmallCodebook.rows();

    __m256i heap_top;
    // whether there is a new dist less than heap_top
    int cmp_mask = 0;

    if constexpr (std::is_same<TargetDType, uint8_t>::value) {
      stepSize = 32;
      heap_top = _mm256_set1_epi8(0xFF);
    } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
      stepSize = 16;
      heap_top = _mm256_set1_epi16(0xFFFF);
    } else {
      assert(false);
    }

    for (int i = 0; i < codebookRow - codebookRow % stepSize; i += stepSize) {

        __m256i acc = _mm256_setzero_si256();

        for (int col = 0; col < numSubspaceNum; ++col) {
            if constexpr (std::is_same<TargetDType, uint8_t>::value) {
                if constexpr(numCentroid == 16) {
                  __m256i code_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + codebookRow*col));
                  __m256i dist_vector = _mm256_shuffle_epi8(lut_registers[col], code_vector); 
                  acc = _mm256_adds_epu8(acc, dist_vector);
                } else if constexpr (numCentroid == 32) {
                  // Each code is 8 bits , which can be represented as 0x0Y or 0x1Y, X is high 4 bits, Y is low 4 bits
                  const __m256i mask_low4 = _mm256_set1_epi8(0x0F);
                  //const __m256i mask_high16 = _mm256_set1_epi8(0x10);

                  __m256i lut_low = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x00); // dist of first 16 centroids of lut, copy twice to 256 bits
                  __m256i lut_high = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x11); // dist of last 16 centroids of lut, copy twice to 256 bits

                  __m256i code_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + codebookRow*col)); // 32 codes, 8 bits each

                  __m256i indices_low = _mm256_and_si256(code_vector, mask_low4); // byte-wise AND, each 1 byte code only get its low 4 bits (0xXY -> 0x0Y)
                  __m256i result_low = _mm256_shuffle_epi8(lut_low, indices_low); // look-up based on 0x0Y

                  // 32*8 bits mask, each 8 bits represent whether this code is 0x0Y or 0x1Y
                  __m256i cmp = _mm256_cmpgt_epi8(code_vector, mask_low4); 
                  __m256i result_high = _mm256_shuffle_epi8(lut_high, indices_low); // look-up based on 0x1Y

                  // select result_low if the code is 0x0Y, select result_high if the code is 0x1Y
                  __m256i dist_vector = _mm256_blendv_epi8(result_low, result_high, cmp); // conditional selection

                  acc = _mm256_adds_epu8(acc, dist_vector);
                }
            } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
              // only AVX512 support _mm256_permute16x16_si256
              // so we need to use _mm256_permute8x32_si256 instead for AVX2, then why not use uint32
              assert(false && "NOT IMPLEMENTED!");
            }
        }

        codes += stepSize;
        __m256i cmp_result;
        if constexpr (std::is_same<TargetDType, uint8_t>::value) {
          // cmp_result = _mm256_cmpgt_epi8(heap_top, acc); // avx2 not supoort uint8 cmp

          __m256i min_u = _mm256_min_epu8(acc, heap_top);
          __m256i acc_le_heaptop = _mm256_cmpeq_epi8(min_u, acc); // if a pair is equal, set to 1s
          cmp_mask = _mm256_movemask_epi8(acc_le_heaptop); // if there is any pair that acc[i] == min(acc[i], heap_top) => new dist found
        } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
          assert(false && "NOT IMPLEMENTED!");
        }
        
        if(cmp_mask){
          alignas(32) TargetDType temp[32];
          _mm256_store_si256(reinterpret_cast<__m256i*>(temp), acc);
          for (int j = 0; j < stepSize; ++j) {
              TargetDType dist = temp[j];
              if (dist < res.worst()) { 
                  res.push(dist, ID{idBias,  i + j});
              }
          }

          if constexpr (std::is_same<TargetDType, uint8_t>::value) {
            heap_top = _mm256_set1_epi8(res.worst());
          } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
            assert(false && "NOT IMPLEMENTED!");
          }
        }
    }

    for(int i = codebookRow - codebookRow % stepSize; i < codebookRow; i++) {
      TargetDType dist = 0;
      const TargetDType * luts = lut.data();
      const int ksub = lut.rows();
  
      for (int col=0; col < mSubspaceNum; col++) {
        dist = unsigendSaturatedAdd<TargetDType>(dist, luts[*codes++]);
        luts += ksub;
      }
      
      if (dist < res.worst()) {        
        res.push(dist, ID{idBias,  i});         
      }
    }

    //f::heap_reorder<f::CMax<TargetDType, int>>(k, heap_dis, heap_ids);
}


// gather can only look up 8 int32 indices in memory, bug partitioned shuffle and merge can look up 32 uint8 once
template <typename TargetDType, int numCentroid, int numSubspaceNum>
void PQ::searchHeapSIMDLargeCentroids(ColMatrix<TargetDType>& lut, const int k, int q_idx,
                        TopKHeap& res) {


    const int ksub = lut.rows();    

    __m256i lut_registers[numSubspaceNum*2];
    for (int col = 0; col < 2*numSubspaceNum;) {
        lut_registers[col] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/2) * ksub));
        lut_registers[col + 1] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/2) * ksub + 32 ));
        col += 2;
    }

    uint8_t* codes = mSmallCodebook.data();

    int stepSize;
    const int codebookRow = mSmallCodebook.rows();

    __m256i heap_top;
    // whether there is a new dist less than heap_top
    int cmp_mask = 0;

    if constexpr (std::is_same<TargetDType, uint8_t>::value) {
      stepSize = 32;
      heap_top = _mm256_set1_epi8(0xFF);
    } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
      stepSize = 16;
      heap_top = _mm256_set1_epi16(0xFFFF);
    } else {
      assert(false);
    }

    for (int i = 0; i < codebookRow - codebookRow % stepSize; i += stepSize) {

        __m256i acc = _mm256_setzero_si256(); 

        for (int col = 0; col < numSubspaceNum*2;) {
            if constexpr (std::is_same<TargetDType, uint8_t>::value) {
                if constexpr (numCentroid == 64) {
                  const __m256i mask_0 = _mm256_set1_epi8(0x0F);
                  const __m256i mask_1 = _mm256_set1_epi8(0x1F);
                  const __m256i mask_2 = _mm256_set1_epi8(0x2F);


                  __m256i lut_0 = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x00);
                  __m256i lut_1 = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x11);
                  __m256i lut_2 = _mm256_permute2x128_si256(lut_registers[col + 1], lut_registers[col + 1], 0x00);
                  __m256i lut_3 = _mm256_permute2x128_si256(lut_registers[col + 1], lut_registers[col + 1], 0x11);

                  __m256i code_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + codebookRow*(col/2))); // 32 codes, 8 bits each
                  __m256i indices_low = _mm256_and_si256(code_vector, mask_0); // byte-wise AND, each 1 byte code only get its low 4 bits (0xXY -> 0x0Y)

                  __m256i result_0 = _mm256_shuffle_epi8(lut_0, indices_low); // look-up based on 0x0Y
                  __m256i result_1 = _mm256_shuffle_epi8(lut_1, indices_low); // look-up based on 0x1Y
                  
                  // select lut_0 if code is 0x0Y. select lut_1 if code is 0x1Y ~ 0x3Y
                  __m256i cmp = _mm256_cmpgt_epi8(code_vector, mask_0); 
                  __m256i result_01 = _mm256_blendv_epi8(result_0, result_1, cmp); // conditional selection

                  __m256i result_2 = _mm256_shuffle_epi8(lut_2, indices_low); // look-up based on 0x2Y
                  __m256i result_3 = _mm256_shuffle_epi8(lut_3, indices_low); // look-up based on 0x3Y
                  
                  // select lut_3 if code is 0x3Y. select lut_2 if code is 0x0Y ~ 0x2Y
                  cmp = _mm256_cmpgt_epi8(code_vector, mask_2); 
                  __m256i result_23 = _mm256_blendv_epi8(result_2, result_3, cmp); // conditional selection

                  // select result_01 if code is 0x0Y or 0x1Y. select result_23 if code is 0x2Y or 0x3Y
                  cmp = _mm256_cmpgt_epi8(code_vector, mask_1); 
                  __m256i dist_vector = _mm256_blendv_epi8(result_01, result_23, cmp);

                  acc = _mm256_adds_epu8(acc, dist_vector);

                  col += 2;
                //   if(col == 24){
                //     __m256i diff = _mm256_subs_epu8(acc, heap_top);           // saturated acc - heap
                //     static __m256i zero = _mm256_setzero_si256(); 
                //     __m256i cmp = _mm256_cmpeq_epi8(diff, zero);
                //     int mask = _mm256_movemask_epi8(cmp);
                //     if(!mask){ // no zero → all acc[i] > heap_top[i]
                //       goto done;
                //     }
                //   }
                }
            } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
              // only AVX512 support _mm256_permute16x16_si256
              // so we need to use _mm256_permute8x32_si256 instead for AVX2, then why not use uint32
              assert(false && "NOT IMPLEMENTED!");
            }
        }

        
        if constexpr (std::is_same<TargetDType, uint8_t>::value) {

          __m256i min_u = _mm256_min_epu8(acc, heap_top);
          __m256i acc_le_heaptop = _mm256_cmpeq_epi8(min_u, acc); // if a pair is equal, set to 1s
          cmp_mask = _mm256_movemask_epi8(acc_le_heaptop); // if there is any pair that acc[i] == min(acc[i], heap_top) => new dist found
        } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
          assert(false && "NOT IMPLEMENTED!");
        }
        
        if(cmp_mask){
          alignas(32) TargetDType temp[32];
          _mm256_store_si256(reinterpret_cast<__m256i*>(temp), acc);
          for (int j = 0; j < stepSize; ++j) {
              TargetDType dist = temp[j];
              if (dist < res.worst()) { 
                  res.push(dist, ID{idBias,  i + j});
              }
          }

          if constexpr (std::is_same<TargetDType, uint8_t>::value) {
            heap_top = _mm256_set1_epi8(res.worst());
          } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
            assert(false && "NOT IMPLEMENTED!");
          }
        }

// done:
        codes += stepSize;
    }

    for(int i = codebookRow - codebookRow % stepSize; i < codebookRow; i++) {
      TargetDType dist = 0;
      const TargetDType * luts = lut.data();
      const int ksub = lut.rows();
  
      for (int col=0; col < mSubspaceNum; col++) {
        dist = unsigendSaturatedAdd<TargetDType>(dist, luts[*codes++]);
        luts += ksub;
      }
      
      if (dist < res.worst()) {        
        res.push(dist, ID{idBias,  i});         
      }
    }

    //f::heap_reorder<f::CMax<TargetDType, int>>(k, heap_dis, heap_ids);
}

static inline bool all_u8_lt_64(__m256i x) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);          // -128
    const __m256i thr  = _mm256_set1_epi8((char)(64 - 128));    // -64

    __m256i xs = _mm256_xor_si256(x, bias);                     // unsigned -> signed domain
    __m256i m  = _mm256_cmpgt_epi8(thr, xs);                    // (xs < thr) => 0xFF else 0x00

    // movemask: takes MSB of each byte => 1 iff that byte is 0xFF
    return (uint32_t)_mm256_movemask_epi8(m) == 0xFFFFFFFFu;
}

static __m256i load32(const std::vector<uint8_t>& v) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v.data()));
}


template <typename TargetDType, int numSubspaceNum>
void PQ::searchHeapSIMD128Centroids(
  ColMatrix<TargetDType>& lut, const int k, int q_idx,
  TopKHeap& res) {

    constexpr int numCentroid=128;

    const int ksub = lut.rows();    

    __m256i lut_registers[numSubspaceNum*4];
    for (int col = 0; col < 4*numSubspaceNum;) {
        lut_registers[col] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/4) * ksub));
        lut_registers[col + 1] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/4) * ksub + 32 ));
        lut_registers[col + 2] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/4) * ksub + 64 ));
        lut_registers[col + 3] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/4) * ksub + 96 ));
        col += 4;
    }

    // mSmallCodebook is col-major
    uint8_t* codes = mSmallCodebook.data();


    int stepSize;
    const int codebookRow = mSmallCodebook.rows();

    __m256i heap_top;
    // whether there is a new dist less than heap_top
    int cmp_mask = 0;

    if constexpr (std::is_same<TargetDType, uint8_t>::value) {
      stepSize = 32;
      heap_top = _mm256_set1_epi8(res.worst());
    } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
      stepSize = 16;
      heap_top = _mm256_set1_epi16(0xFFFF);
    } else {
      assert(false);
    }

    long long* prunePtr = pruneMarks.data();
    long long curPrune  = *prunePtr++; 
    uint8_t prunePos = 0;

    const __m256i mask_0 = _mm256_set1_epi8(0x0F);
    const __m256i mask_1 = _mm256_set1_epi8(0x1F);
    const __m256i mask_2 = _mm256_set1_epi8(0x2F);
    const __m256i mask_3 = _mm256_set1_epi8(0x3F);
    const __m256i mask_4 = _mm256_set1_epi8(0x4F);
    const __m256i mask_5 = _mm256_set1_epi8(0x5F);
    const __m256i mask_6 = _mm256_set1_epi8(0x6F);


    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i mask_hi3 = _mm256_set1_epi8(0x07);
    const __m256i zero     = _mm256_setzero_si256();
    const __m256i ones     = _mm256_set1_epi8((char)0xFF);

    struct LUTPair{
      __m256i lo, hi;
    };

    alignas(32) LUTPair lutPairs[numSubspaceNum*4];    

    //auto start = std::chrono::steady_clock::now();
    for (int col = 0; col < numSubspaceNum*4; col += 4){
      lutPairs[col].lo = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x00);
      lutPairs[col].hi = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x11);
      lutPairs[col + 1].lo = _mm256_permute2x128_si256(lut_registers[col + 1], lut_registers[col + 1], 0x00);
      lutPairs[col + 1].hi = _mm256_permute2x128_si256(lut_registers[col + 1], lut_registers[col + 1], 0x11);
      lutPairs[col + 2].lo = _mm256_permute2x128_si256(lut_registers[col + 2], lut_registers[col + 2], 0x00);
      lutPairs[col + 2].hi = _mm256_permute2x128_si256(lut_registers[col + 2], lut_registers[col + 2], 0x11);
      lutPairs[col + 3].lo = _mm256_permute2x128_si256(lut_registers[col + 3], lut_registers[col + 3], 0x00);
      lutPairs[col + 3].hi = _mm256_permute2x128_si256(lut_registers[col + 3], lut_registers[col + 3], 0x11);
    }

    // auto end = std::chrono::steady_clock::now();
    // keyKernalSec += std::chrono::duration<double>(end - start).count();


    for (int i = 0; i < codebookRow - codebookRow % stepSize; i += stepSize) {

        __m256i acc = _mm256_setzero_si256();

        for (int col = 0; col < numSubspaceNum*4;) {
            if constexpr (std::is_same<TargetDType, uint8_t>::value) {
                // 批量加载 32 个 codes
                if constexpr (numCentroid == 128) {
                  __m256i code_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + codebookRow*(col/4))); // 32 codes, 8 bits each



                  //if (!isPrune) {
                  if (true) {

                    __m256i indices_low = _mm256_and_si256(code_vector, mask_0); // byte-wise AND, each 1 byte code only get its low 4 bits (0xXY -> 0x0Y)

                    __m256i result_0 = _mm256_shuffle_epi8(lutPairs[col].lo, indices_low); // look-up based on 0x0Y
                    __m256i result_1 = _mm256_shuffle_epi8(lutPairs[col].hi, indices_low); // look-up based on 0x1Y
                    
                    // select lut_0 if code is 0x0Y. select lut_1 if code is 0x1Y ~ 0x3Y
                    __m256i cmp = _mm256_cmpgt_epi8(code_vector, mask_0); 
                    __m256i result_01 = _mm256_blendv_epi8(result_0, result_1, cmp); // conditional selection

                    __m256i result_2 = _mm256_shuffle_epi8(lutPairs[col+1].lo, indices_low); // look-up based on 0x2Y
                    __m256i result_3 = _mm256_shuffle_epi8(lutPairs[col+1].hi, indices_low); // look-up based on 0x3Y
                    
                    // select lut_3 if code is 0x3Y. select lut_2 if code is 0x0Y ~ 0x2Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_2); 
                    __m256i result_23 = _mm256_blendv_epi8(result_2, result_3, cmp); // conditional selection

                    // select result_01 if code is 0x0Y or 0x1Y. select result_23 if code is 0x2Y or 0x3Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_1); 
                    __m256i result_0123 = _mm256_blendv_epi8(result_01, result_23, cmp);

                    __m256i result_4 = _mm256_shuffle_epi8(lutPairs[col+2].lo, indices_low); // look-up based on 0x0Y
                    __m256i result_5 = _mm256_shuffle_epi8(lutPairs[col+2].hi, indices_low); // look-up based on 0x1Y
                    
                    // select lut_4 if code is 0x0Y. select lut_5 if code is 0x1Y ~ 0x7Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_4); 
                    __m256i result_45 = _mm256_blendv_epi8(result_4, result_5, cmp); // conditional selection

                    __m256i result_6 = _mm256_shuffle_epi8(lutPairs[col+3].lo, indices_low); // look-up based on 0x2Y
                    __m256i result_7 = _mm256_shuffle_epi8(lutPairs[col+3].hi, indices_low); // look-up based on 0x3Y

                    cmp = _mm256_cmpgt_epi8(code_vector, mask_6); 
                    __m256i result_67 = _mm256_blendv_epi8(result_6, result_7, cmp); // conditional selection
                    
                    // select lut_3 if code is 0x3Y. select lut_2 if code is 0x0Y ~ 0x2Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_5); 
                    __m256i result_4567 = _mm256_blendv_epi8(result_45, result_67, cmp); // conditional selection

                    // select result_01 if code is 0x0Y or 0x1Y. select result_23 if code is 0x2Y or 0x3Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_3); 
                    __m256i dist = _mm256_blendv_epi8(result_0123, result_4567, cmp);

                    acc = _mm256_adds_epu8(acc, dist);
                    col += 4;
                  } else {
                    __m256i indices_low = _mm256_and_si256(code_vector, mask_0); // byte-wise AND, each 1 byte code only get its low 4 bits (0xXY -> 0x0Y)

                    __m256i result_0 = _mm256_shuffle_epi8(lutPairs[col].lo, indices_low); // look-up based on 0x0Y
                    __m256i result_1 = _mm256_shuffle_epi8(lutPairs[col].hi, indices_low); // look-up based on 0x1Y
                    
                    // select lut_0 if code is 0x0Y. select lut_1 if code is 0x1Y ~ 0x3Y
                    __m256i cmp = _mm256_cmpgt_epi8(code_vector, mask_0); 
                    __m256i result_01 = _mm256_blendv_epi8(result_0, result_1, cmp); // conditional selection

                    __m256i result_2 = _mm256_shuffle_epi8(lutPairs[col+1].lo, indices_low); // look-up based on 0x2Y
                    __m256i result_3 = _mm256_shuffle_epi8(lutPairs[col+1].hi, indices_low); // look-up based on 0x3Y
                    
                    // select lut_3 if code is 0x3Y. select lut_2 if code is 0x0Y ~ 0x2Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_2); 
                    __m256i result_23 = _mm256_blendv_epi8(result_2, result_3, cmp); // conditional selection

                    // select result_01 if code is 0x0Y or 0x1Y. select result_23 if code is 0x2Y or 0x3Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_1); 
                    __m256i dist_vector = _mm256_blendv_epi8(result_01, result_23, cmp);

                    acc = _mm256_adds_epu8(acc, dist_vector);
                    col += 4;
                  }

                }
            } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
              // only AVX512 support _mm256_permute16x16_si256
              // so we need to use _mm256_permute8x32_si256 instead for AVX2, then why not use uint32
              assert(false && "NOT IMPLEMENTED!");
            }


        }

        codes += stepSize;
        __m256i cmp_result;
        if constexpr (std::is_same<TargetDType, uint8_t>::value) {
          // cmp_result = _mm256_cmpgt_epi8(heap_top, acc); // avx2 not supoort uint8 cmp, only int8 cmp

          // __m256i min_u = _mm256_min_epu8(acc, heap_top);
          // __m256i acc_le_heaptop = _mm256_cmpeq_epi8(min_u, acc); // if a pair is equal, set to 1s
          // cmp_mask = _mm256_movemask_epi8(acc_le_heaptop); // if there is any pair that acc[i] == min(acc[i], heap_top) => new dist found
          
          
          __m256i flip = _mm256_set1_epi8(char(0x80));
          __m256i a = _mm256_xor_si256(acc, flip);
          __m256i b = _mm256_xor_si256(heap_top, flip);
          __m256i lt = _mm256_cmpgt_epi8(b, a);          // (b > a) signed  <=> acc < heap_top unsigned
          cmp_mask = _mm256_movemask_epi8(lt);

          // __m256i d = _mm256_subs_epu8(heap_top, acc);
          // cmp_mask = _mm256_movemask_epi8(_mm256_cmpgt_epi8(d, _mm256_setzero_si256()));

        } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
          assert(false && "NOT IMPLEMENTED!");
        }

        
        
        if(cmp_mask){
          alignas(32) uint8_t temp[32];
          _mm256_store_si256(reinterpret_cast<__m256i*>(temp), acc);

          while (cmp_mask) {
            int j = __builtin_ctz(cmp_mask);  
            res.push(temp[j], ID{idBias,  i + j});
            cmp_mask &= cmp_mask - 1;                
          }

          heap_top = _mm256_set1_epi8(res.worst());
          //keyKernalSec++;

        }
        

    }

    for(int i = codebookRow - codebookRow % stepSize; i < codebookRow; i++) {
      TargetDType dist = 0;
      const TargetDType * luts = lut.data();
      const int ksub = lut.rows();
  
      for (int col=0; col < mSubspaceNum; col++) {
        dist = unsigendSaturatedAdd<TargetDType>(dist, luts[*codes++]);
        luts += ksub;
      }
      
      if (dist < res.worst()) {        
        res.push(dist,  ID{idBias,  i});         
      }

    }

    //f::heap_reorder<f::CMax<TargetDType, int>>(k, heap_dis, heap_ids);
}




template<typename TargetDType>
LabelDistVecF PQ::searchOne(const RowVectorXf &XTest, const int k, bool simd, bool verbose,
  TopKHeap &answers, int idBias, bool retrainQuant) {
  this->idBias = idBias;
  
  CreateLUT(XTest);

  LabelDistVecF ret;
  ret.labels.resize(k);
  ret.distances.resize(k);

  if constexpr (!std::is_same<TargetDType, float>::value) {
    std::vector<TargetDType> distances;
    distances.resize(k);
    //f::HeapArray<f::CMax<TargetDType, int>> answers = { 1, size_t(k), ret.labels.data(), distances.data() };
    

    static DataQuantizer quantizer;
    static ColMatrix<TargetDType> quantizedLUT;
    if(retrainQuant) quantizer.train(lut, mSmallCodebook, true);
    quantizedLUT = quantizer.quantize<TargetDType>(lut);
    //start = std::chrono::high_resolution_clock::now();
  
    if(!simd){
      searchHeap<TargetDType>(quantizedLUT, k, 0, answers);
    } else {
      if constexpr (std::is_same<TargetDType, uint8_t>::value) {
        if (mCentroidsNum == 128) {
          switch(mSubspaceNum){
            case 2: searchHeapSIMD128Centroids<uint8_t, 2>(quantizedLUT, k, 0, answers); break;
            case 3: searchHeapSIMD128Centroids<uint8_t, 3>(quantizedLUT, k, 0, answers); break;
            case 4: searchHeapSIMD128Centroids<uint8_t, 4>(quantizedLUT, k, 0, answers); break;
            case 6: searchHeapSIMD128Centroids<uint8_t, 6>(quantizedLUT, k, 0, answers); break;
            case 8: searchHeapSIMD128Centroids<uint8_t, 8>(quantizedLUT, k, 0, answers); break;
            case 10: searchHeapSIMD128Centroids<uint8_t, 10>(quantizedLUT, k, 0, answers); break;
            case 11: searchHeapSIMD128Centroids<uint8_t, 11>(quantizedLUT, k, 0, answers); break;
            case 12: searchHeapSIMD128Centroids<uint8_t, 12>(quantizedLUT, k, 0, answers); break;
            case 13: searchHeapSIMD128Centroids<uint8_t, 13>(quantizedLUT, k, 0, answers); break;
            case 14: searchHeapSIMD128Centroids<uint8_t, 14>(quantizedLUT, k, 0, answers); break;
            case 15: searchHeapSIMD128Centroids<uint8_t, 15>(quantizedLUT, k, 0, answers); break;
            case 16: searchHeapSIMD128Centroids<uint8_t, 16>(quantizedLUT, k, 0, answers); break;
            case 18: searchHeapSIMD128Centroids<uint8_t, 18>(quantizedLUT, k, 0, answers); break;
            case 20: searchHeapSIMD128Centroids<uint8_t, 20>(quantizedLUT, k, 0, answers); break;
            case 22: searchHeapSIMD128Centroids<uint8_t, 22>(quantizedLUT, k, 0, answers); break;
            case 24: searchHeapSIMD128Centroids<uint8_t, 24>(quantizedLUT, k, 0, answers); break;
            case 26: searchHeapSIMD128Centroids<uint8_t, 26>(quantizedLUT, k, 0, answers); break;
            case 28: searchHeapSIMD128Centroids<uint8_t, 28>(quantizedLUT, k, 0, answers); break;
            case 30: searchHeapSIMD128Centroids<uint8_t, 30>(quantizedLUT, k, 0, answers); break;
            case 32: searchHeapSIMD128Centroids<uint8_t, 32>(quantizedLUT, k, 0, answers); break;
            case 36: searchHeapSIMD128Centroids<uint8_t, 36>(quantizedLUT, k, 0, answers); break;
            case 64: searchHeapSIMD128Centroids<uint8_t, 64>(quantizedLUT, k, 0, answers); break;
            default:
              switch(mSubspaceNum) {
                case 5: searchHeapSIMD128Centroids<uint8_t, 5>(quantizedLUT, k, 0, answers); break;
                case 7: searchHeapSIMD128Centroids<uint8_t, 7>(quantizedLUT, k, 0, answers); break;
                case 9: searchHeapSIMD128Centroids<uint8_t, 9>(quantizedLUT, k, 0, answers); break;
                case 11: searchHeapSIMD128Centroids<uint8_t, 11>(quantizedLUT, k, 0, answers); break;
                case 13: searchHeapSIMD128Centroids<uint8_t, 13>(quantizedLUT, k, 0, answers); break;
                case 15: searchHeapSIMD128Centroids<uint8_t, 15>(quantizedLUT, k, 0, answers); break;
                case 17: searchHeapSIMD128Centroids<uint8_t, 17>(quantizedLUT, k, 0, answers); break;
                assert(false && "Unsupported #Subspaces");
              } 
              
          }
        }
        else if (mCentroidsNum == 64) {
          switch(mSubspaceNum){
            case 2: searchHeapSIMDLargeCentroids<uint8_t, 64, 2>(quantizedLUT, k, 0, answers); break;
            case 3: searchHeapSIMDLargeCentroids<uint8_t, 64, 3>(quantizedLUT, k, 0, answers); break;
            case 4: searchHeapSIMDLargeCentroids<uint8_t, 64, 4>(quantizedLUT, k, 0, answers); break;
            case 6: searchHeapSIMDLargeCentroids<uint8_t, 64, 6>(quantizedLUT, k, 0, answers); break;
            case 8: searchHeapSIMDLargeCentroids<uint8_t, 64, 8>(quantizedLUT, k, 0, answers); break;
            case 10: searchHeapSIMDLargeCentroids<uint8_t, 64, 10>(quantizedLUT, k, 0, answers); break;
            case 11: searchHeapSIMDLargeCentroids<uint8_t, 64, 11>(quantizedLUT, k, 0, answers); break;
            case 12: searchHeapSIMDLargeCentroids<uint8_t, 64, 12>(quantizedLUT, k, 0, answers); break;
            case 13: searchHeapSIMDLargeCentroids<uint8_t, 64, 13>(quantizedLUT, k, 0, answers); break;
            case 14: searchHeapSIMDLargeCentroids<uint8_t, 64, 14>(quantizedLUT, k, 0, answers); break;
            case 15: searchHeapSIMDLargeCentroids<uint8_t, 64, 15>(quantizedLUT, k, 0, answers); break;
            case 16: searchHeapSIMDLargeCentroids<uint8_t, 64, 16>(quantizedLUT, k, 0, answers); break;
            case 18: searchHeapSIMDLargeCentroids<uint8_t, 64, 18>(quantizedLUT, k, 0, answers); break;
            case 20: searchHeapSIMDLargeCentroids<uint8_t, 64, 20>(quantizedLUT, k, 0, answers); break;
            case 22: searchHeapSIMDLargeCentroids<uint8_t, 64, 22>(quantizedLUT, k, 0, answers); break;
            case 24: searchHeapSIMDLargeCentroids<uint8_t, 64, 24>(quantizedLUT, k, 0, answers); break;
            case 26: searchHeapSIMDLargeCentroids<uint8_t, 64, 26>(quantizedLUT, k, 0, answers); break;
            case 28: searchHeapSIMDLargeCentroids<uint8_t, 64, 28>(quantizedLUT, k, 0, answers); break;
            case 30: searchHeapSIMDLargeCentroids<uint8_t, 64, 30>(quantizedLUT, k, 0, answers); break;
            case 32: searchHeapSIMDLargeCentroids<uint8_t, 64, 32>(quantizedLUT, k, 0, answers); break;
            case 64: searchHeapSIMDLargeCentroids<uint8_t, 64, 64>(quantizedLUT, k, 0, answers); break;
            default:
              switch(mSubspaceNum) {
                case 5: searchHeapSIMDLargeCentroids<uint8_t, 64, 5>(quantizedLUT, k, 0, answers); break;
                case 7: searchHeapSIMDLargeCentroids<uint8_t, 64, 7>(quantizedLUT, k, 0, answers); break;
                case 9: searchHeapSIMDLargeCentroids<uint8_t, 64, 9>(quantizedLUT, k, 0, answers); break;
                case 11: searchHeapSIMDLargeCentroids<uint8_t, 64, 11>(quantizedLUT, k, 0, answers); break;
                case 13: searchHeapSIMDLargeCentroids<uint8_t, 64, 13>(quantizedLUT, k, 0, answers); break;
                case 15: searchHeapSIMDLargeCentroids<uint8_t, 64, 15>(quantizedLUT, k, 0, answers); break;
                case 17: searchHeapSIMDLargeCentroids<uint8_t, 64, 17>(quantizedLUT, k, 0, answers); break;
                assert(false && "Unsupported #Subspaces");
              } 
          }
        }
        else if(mCentroidsNum == 32) {
          switch(mSubspaceNum){
            case 2: searchHeapSIMD<uint8_t, 32, 2>(quantizedLUT, k, 0, answers); break;
            case 3: searchHeapSIMD<uint8_t, 32, 3>(quantizedLUT, k, 0, answers); break;
            case 4: searchHeapSIMD<uint8_t, 32, 4>(quantizedLUT, k, 0, answers); break;
            case 6: searchHeapSIMD<uint8_t, 32, 6>(quantizedLUT, k, 0, answers); break;
            case 8: searchHeapSIMD<uint8_t, 32, 8>(quantizedLUT, k, 0, answers); break;
            case 10: searchHeapSIMD<uint8_t, 32, 10>(quantizedLUT, k, 0, answers); break;
            case 11: searchHeapSIMD<uint8_t, 32, 11>(quantizedLUT, k, 0, answers); break;
            case 12: searchHeapSIMD<uint8_t, 32, 12>(quantizedLUT, k, 0, answers); break;
            case 13: searchHeapSIMD<uint8_t, 32, 13>(quantizedLUT, k, 0, answers); break;
            case 14: searchHeapSIMD<uint8_t, 32, 14>(quantizedLUT, k, 0, answers); break;
            case 15: searchHeapSIMD<uint8_t, 32, 15>(quantizedLUT, k, 0, answers); break;
            case 16: searchHeapSIMD<uint8_t, 32, 16>(quantizedLUT, k, 0, answers); break;
            case 18: searchHeapSIMD<uint8_t, 32, 18>(quantizedLUT, k, 0, answers); break;
            case 20: searchHeapSIMD<uint8_t, 32, 20>(quantizedLUT, k, 0, answers); break;
            case 22: searchHeapSIMD<uint8_t, 32, 22>(quantizedLUT, k, 0, answers); break;
            case 24: searchHeapSIMD<uint8_t, 32, 24>(quantizedLUT, k, 0, answers); break;
            case 26: searchHeapSIMD<uint8_t, 32, 26>(quantizedLUT, k, 0, answers); break;
            case 28: searchHeapSIMD<uint8_t, 32, 28>(quantizedLUT, k, 0, answers); break;
            case 30: searchHeapSIMD<uint8_t, 32, 30>(quantizedLUT, k, 0, answers); break;
            case 32: searchHeapSIMD<uint8_t, 32, 32>(quantizedLUT, k, 0, answers); break;
            case 64: searchHeapSIMD<uint8_t, 32, 64>(quantizedLUT, k, 0, answers); break;
            default:
              switch(mSubspaceNum) {
                case 5: searchHeapSIMD<uint8_t, 32, 5>(quantizedLUT, k, 0, answers); break;
                case 7: searchHeapSIMD<uint8_t, 32, 7>(quantizedLUT, k, 0, answers); break;
                case 9: searchHeapSIMD<uint8_t, 32, 9>(quantizedLUT, k, 0, answers); break;
                case 11: searchHeapSIMD<uint8_t, 32, 11>(quantizedLUT, k, 0, answers); break;
                case 13: searchHeapSIMD<uint8_t, 32, 13>(quantizedLUT, k, 0, answers); break;
                case 15: searchHeapSIMD<uint8_t, 32, 15>(quantizedLUT, k, 0, answers); break;
                case 17: searchHeapSIMD<uint8_t, 32, 17>(quantizedLUT, k, 0, answers); break;
                assert(false && "Unsupported #Subspaces");
              } 
          }
        }
        else if(mCentroidsNum == 16) {
          switch(mSubspaceNum){
            case 2: searchHeapSIMD<uint8_t, 16, 2>(quantizedLUT, k, 0, answers); break;
            case 3: searchHeapSIMD<uint8_t, 16, 3>(quantizedLUT, k, 0, answers); break;
            case 4: searchHeapSIMD<uint8_t, 16, 4>(quantizedLUT, k, 0, answers); break;
            case 6: searchHeapSIMD<uint8_t, 16, 6>(quantizedLUT, k, 0, answers); break;
            case 8: searchHeapSIMD<uint8_t, 16, 8>(quantizedLUT, k, 0, answers); break;
            case 10: searchHeapSIMD<uint8_t, 16, 10>(quantizedLUT, k, 0, answers); break;
            case 11: searchHeapSIMD<uint8_t, 16, 11>(quantizedLUT, k, 0, answers); break;
            case 12: searchHeapSIMD<uint8_t, 16, 12>(quantizedLUT, k, 0, answers); break;
            case 13: searchHeapSIMD<uint8_t, 16, 13>(quantizedLUT, k, 0, answers); break;
            case 14: searchHeapSIMD<uint8_t, 16, 14>(quantizedLUT, k, 0, answers); break;
            case 15: searchHeapSIMD<uint8_t, 16, 15>(quantizedLUT, k, 0, answers); break;
            case 16: searchHeapSIMD<uint8_t, 16, 16>(quantizedLUT, k, 0, answers); break;
            case 18: searchHeapSIMD<uint8_t, 16, 18>(quantizedLUT, k, 0, answers); break;
            case 20: searchHeapSIMD<uint8_t, 16, 20>(quantizedLUT, k, 0, answers); break;
            case 22: searchHeapSIMD<uint8_t, 16, 22>(quantizedLUT, k, 0, answers); break;
            case 24: searchHeapSIMD<uint8_t, 16, 24>(quantizedLUT, k, 0, answers); break;
            case 26: searchHeapSIMD<uint8_t, 16, 26>(quantizedLUT, k, 0, answers); break;
            case 28: searchHeapSIMD<uint8_t, 16, 28>(quantizedLUT, k, 0, answers); break;
            case 30: searchHeapSIMD<uint8_t, 16, 30>(quantizedLUT, k, 0, answers); break;
            case 32: searchHeapSIMD<uint8_t, 16, 32>(quantizedLUT, k, 0, answers); break;
            case 64: searchHeapSIMD<uint8_t, 16, 64>(quantizedLUT, k, 0, answers); break;
            default:
              switch(mSubspaceNum) {
                case 5: searchHeapSIMD<uint8_t, 16, 5>(quantizedLUT, k, 0, answers); break;
                case 7: searchHeapSIMD<uint8_t, 16, 7>(quantizedLUT, k, 0, answers); break;
                case 9: searchHeapSIMD<uint8_t, 16, 9>(quantizedLUT, k, 0, answers); break;
                case 11: searchHeapSIMD<uint8_t, 16, 11>(quantizedLUT, k, 0, answers); break;
                case 13: searchHeapSIMD<uint8_t, 16, 13>(quantizedLUT, k, 0, answers); break;
                case 15: searchHeapSIMD<uint8_t, 16, 15>(quantizedLUT, k, 0, answers); break;
                case 17: searchHeapSIMD<uint8_t, 16, 17>(quantizedLUT, k, 0, answers); break;
                assert(false && "Unsupported #Subspaces");
              } 
          }
        } else {
          assert(false && "Unsupported Config");
        }
      } else if constexpr(std::is_same<TargetDType, uint16_t>::value) {
          assert(mCentroidsNum == 16 && "Only support 16*uint16=32B for avx2");
          switch(mSubspaceNum){
              case 8: searchHeapSIMD<uint16_t, 16, 8>(quantizedLUT, k, 0, answers); break;
              case 9: searchHeapSIMD<uint16_t, 16, 9>(quantizedLUT, k, 0, answers); break;
              case 10: searchHeapSIMD<uint16_t, 16, 10>(quantizedLUT, k, 0, answers); break;
              case 11: searchHeapSIMD<uint16_t, 16, 11>(quantizedLUT, k, 0, answers); break;
              case 12: searchHeapSIMD<uint16_t, 16, 12>(quantizedLUT, k, 0, answers); break;
              case 13: searchHeapSIMD<uint16_t, 16, 13>(quantizedLUT, k, 0, answers); break;
              case 14: searchHeapSIMD<uint16_t, 16, 14>(quantizedLUT, k, 0, answers); break;
              default:
                assert(false && "Unsupported #Subspaces");
            }
      } else {
        assert(false && "Unsupported Config");
      }
    }

    //ret.distances = quantizer.dequantize<TargetDType>(distances);
  } else {
    assert((!std::is_same<TargetDType, float>::value && simd) && "DType of LUT can only be uint8 or uint16 if simd is enabled");
    // TopKHeap answers(k);
    searchHeap<TargetDType>(lut, k, 0, answers);
  }
  //end = std::chrono::high_resolution_clock::now();
  //kernelSec += std::chrono::duration<double>(end - start).count();

  // double pruntRate = double(pruntCount) / (mSmallCodebook.rows() / 32) / mSubspaceNum;
  // //printf("pruning rate is %lf\n", pruntRate);wers
  // pruntCount = 0;

  // std::copy(answers.ids, answers.ids+k, ret.labels.begin());
  // std::copy(answers.dist, answers.dist, ret.distances.begin());
  //ret.distances.resize(k);


  return ret;
}

template <typename TargetDType, int numCentroids, int numSubspaceNum>
void PQ::searchHeapSIMD(
  ColMatrix<TargetDType>& lut, const int k, int q_idx,
  TopKHeap& res, ColMatrix<uint8_t> &curCodes, std::vector<int>& idmap) {

    const int ksub = lut.rows();   


    __m256i lut_registers[numSubspaceNum]; 
    if constexpr (numCentroids==16){
      for (int col = 0; col < numSubspaceNum; ++col) {
          __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut.data() + col * ksub));
          lut_registers[col] = _mm256_set_m128i(lut128, lut128);
      }
    } else {
      for (int col = 0; col < numSubspaceNum; ++col) {
          lut_registers[col] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + col * ksub));
      }
    }

    // curCodes is col-major
    uint8_t* codes = curCodes.data();


    int stepSize;
    const int codebookRow = curCodes.rows();

    __m256i heap_top;
    // whether there is a new dist less than heap_top
    int cmp_mask = 0;

    if constexpr (std::is_same<TargetDType, uint8_t>::value) {
      stepSize = 32;
      heap_top = _mm256_set1_epi8(res.worst());
    } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
      stepSize = 16;
      heap_top = _mm256_set1_epi16(0xFFFF);
    } else {
      assert(false);
    }


    const __m256i mask_0 = _mm256_set1_epi8(0x0F);

    struct LUTPair{
      __m256i lo, hi;
    };

    alignas(32) LUTPair lutPairs[numSubspaceNum];    
    if constexpr(numCentroids == 32) {
    for (int col = 0; col < numSubspaceNum; col += 1){
        lutPairs[col].lo = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x00);
        lutPairs[col].hi = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x11);
      }
    }

    // auto end = std::chrono::steady_clock::now();
    // keyKernalSec += std::chrono::duration<double>(end - start).count();


    for (int i = 0; i < codebookRow - codebookRow % stepSize; i += stepSize) {

        __m256i acc = _mm256_setzero_si256(); 

        for (int col = 0; col < numSubspaceNum; col++) {
            __m256i code_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + codebookRow*(col))); // 32 codes, 8 bits each
            if constexpr (numCentroids == 32) {

                __m256i indices_low = _mm256_and_si256(code_vector, mask_0); // byte-wise AND, each 1 byte code only get its low 4 bits (0xXY -> 0x0Y)

                __m256i result_0 = _mm256_shuffle_epi8(lutPairs[col].lo, indices_low); // look-up based on 0x0Y
                __m256i result_1 = _mm256_shuffle_epi8(lutPairs[col].hi, indices_low); // look-up based on 0x1Y
                
                // select lut_0 if code is 0x0Y. select lut_1 if code is 0x1Y ~ 0x3Y
                __m256i cmp = _mm256_cmpgt_epi8(code_vector, mask_0); 
                __m256i dist_vector = _mm256_blendv_epi8(result_0, result_1, cmp); // conditional selection

                acc = _mm256_adds_epu8(acc, dist_vector);

            } else if constexpr(numCentroids == 16) {
                __m256i code_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + codebookRow*col));
                __m256i dist_vector = _mm256_shuffle_epi8(lut_registers[col], code_vector); 
                acc = _mm256_adds_epu8(acc, dist_vector);
            }
        }

        __m256i cmp_result;
        if constexpr (std::is_same<TargetDType, uint8_t>::value) {
          // cmp_result = _mm256_cmpgt_epi8(heap_top, acc); // avx2 not supoort uint8 cmp, only int8 cmp

          // __m256i min_u = _mm256_min_epu8(acc, heap_top);
          // __m256i acc_le_heaptop = _mm256_cmpeq_epi8(min_u, acc); // if a pair is equal, set to 1s
          // cmp_mask = _mm256_movemask_epi8(acc_le_heaptop); // if there is any pair that acc[i] == min(acc[i], heap_top) => new dist found
          
          
          __m256i flip = _mm256_set1_epi8(char(0x80));
          __m256i a = _mm256_xor_si256(acc, flip);
          __m256i b = _mm256_xor_si256(heap_top, flip);
          __m256i lt = _mm256_cmpgt_epi8(b, a);          // (b > a) signed  <=> acc < heap_top unsigned
          cmp_mask = _mm256_movemask_epi8(lt);

        } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
          assert(false && "NOT IMPLEMENTED!");
        }

        
        
        if(cmp_mask){
          alignas(32) uint8_t temp[32];
          _mm256_store_si256(reinterpret_cast<__m256i*>(temp), acc);

          while (cmp_mask) {
            int j = __builtin_ctz(cmp_mask);  
            res.push(temp[j], ID{idBias,  idmap[i + j]});
            cmp_mask &= cmp_mask - 1;         
          }

          heap_top = _mm256_set1_epi8(res.worst());
          //keyKernalSec++;

        }

        codes += stepSize;
        

    }

    const int start = codebookRow - (codebookRow % stepSize);
    const int end   = codebookRow;
    const int R    = curCodes.rows();     // list size
    // 1) cache each LUT column base pointer
    std::vector<const TargetDType*> lut_col(mSubspaceNum);
    const TargetDType* lut_base = lut.data();
    for (int col = 0; col < mSubspaceNum; ++col) {
        lut_col[col] = lut_base + col * ksub;
    }

    // 2) codes base pointer (ColMajor)
    const uint8_t* codes_base = curCodes.data();

    for (int i = start; i < end; ++i) {
        uint16_t acc16 = 0;

        // ColMajor: code(i,col) = codes_base[col*R + i]
        #pragma unroll
        for (int col = 0; col < numSubspaceNum; ++col) {
            const uint8_t code = codes_base[col * R + i];
            acc16 += (uint16_t)lut_col[col][code];
        }

        const uint8_t dist8 = (acc16 > 255u) ? 255u : (uint8_t)acc16;

        if (dist8 < res.worst()) {
            res.push(dist8, ID{idBias, idmap[i]});
        }
    }

}

template <typename TargetDType, int numCentroid, int numSubspaceNum>
void PQ::searchHeapSIMDLargeCentroids(
  ColMatrix<TargetDType>& lut, const int k, int q_idx,
  TopKHeap& res, ColMatrix<uint8_t> &curCodes, std::vector<int>& idmap) {

    const int ksub = lut.rows();     

    __m256i lut_registers[numSubspaceNum*2];
    for (int col = 0; col < 2*numSubspaceNum;) {
        lut_registers[col] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/2) * ksub));
        lut_registers[col + 1] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/2) * ksub + 32 ));
        col += 2;
    }

    // curCodes is col-major
    uint8_t* codes = curCodes.data();


    int stepSize;
    const int codebookRow = curCodes.rows();

    __m256i heap_top;
    // whether there is a new dist less than heap_top
    int cmp_mask = 0;

    if constexpr (std::is_same<TargetDType, uint8_t>::value) {
      stepSize = 32;
      heap_top = _mm256_set1_epi8(res.worst());
    } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
      stepSize = 16;
      heap_top = _mm256_set1_epi16(0xFFFF);
    } else {
      assert(false);
    }


    const __m256i mask_0 = _mm256_set1_epi8(0x0F);
    const __m256i mask_1 = _mm256_set1_epi8(0x1F);
    const __m256i mask_2 = _mm256_set1_epi8(0x2F);

    struct LUTPair{
      __m256i lo, hi;
    };

    alignas(32) LUTPair lutPairs[numSubspaceNum*2];    

    //auto start = std::chrono::steady_clock::now();
    for (int col = 0; col < numSubspaceNum*2; col += 2){
      lutPairs[col].lo = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x00);
      lutPairs[col].hi = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x11);
      lutPairs[col + 1].lo = _mm256_permute2x128_si256(lut_registers[col + 1], lut_registers[col + 1], 0x00);
      lutPairs[col + 1].hi = _mm256_permute2x128_si256(lut_registers[col + 1], lut_registers[col + 1], 0x11);
    }

    // auto end = std::chrono::steady_clock::now();
    // keyKernalSec += std::chrono::duration<double>(end - start).count();


    for (int i = 0; i < codebookRow - codebookRow % stepSize; i += stepSize) {

        __m256i acc = _mm256_setzero_si256();

        for (int col = 0; col < numSubspaceNum*2;) {
            if constexpr (std::is_same<TargetDType, uint8_t>::value) {
                __m256i code_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + codebookRow*(col/2))); // 32 codes, 8 bits each

                __m256i indices_low = _mm256_and_si256(code_vector, mask_0); // byte-wise AND, each 1 byte code only get its low 4 bits (0xXY -> 0x0Y)

                __m256i result_0 = _mm256_shuffle_epi8(lutPairs[col].lo, indices_low); // look-up based on 0x0Y
                __m256i result_1 = _mm256_shuffle_epi8(lutPairs[col].hi, indices_low); // look-up based on 0x1Y
                
                // select lut_0 if code is 0x0Y. select lut_1 if code is 0x1Y ~ 0x3Y
                __m256i cmp = _mm256_cmpgt_epi8(code_vector, mask_0); 
                __m256i result_01 = _mm256_blendv_epi8(result_0, result_1, cmp); // conditional selection

                __m256i result_2 = _mm256_shuffle_epi8(lutPairs[col+1].lo, indices_low); // look-up based on 0x2Y
                __m256i result_3 = _mm256_shuffle_epi8(lutPairs[col+1].hi, indices_low); // look-up based on 0x3Y
                
                // select lut_3 if code is 0x3Y. select lut_2 if code is 0x0Y ~ 0x2Y
                cmp = _mm256_cmpgt_epi8(code_vector, mask_2); 
                __m256i result_23 = _mm256_blendv_epi8(result_2, result_3, cmp); // conditional selection

                // select result_01 if code is 0x0Y or 0x1Y. select result_23 if code is 0x2Y or 0x3Y
                cmp = _mm256_cmpgt_epi8(code_vector, mask_1); 
                __m256i dist_vector = _mm256_blendv_epi8(result_01, result_23, cmp);

                acc = _mm256_adds_epu8(acc, dist_vector);
                col += 2;

            } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
              // only AVX512 support _mm256_permute16x16_si256
              // so we need to use _mm256_permute8x32_si256 instead for AVX2, then why not use uint32
              assert(false && "NOT IMPLEMENTED!");
            }

        }

        codes += stepSize;
        __m256i cmp_result;
        if constexpr (std::is_same<TargetDType, uint8_t>::value) {
          // cmp_result = _mm256_cmpgt_epi8(heap_top, acc); // avx2 not supoort uint8 cmp, only int8 cmp

          // __m256i min_u = _mm256_min_epu8(acc, heap_top);
          // __m256i acc_le_heaptop = _mm256_cmpeq_epi8(min_u, acc); // if a pair is equal, set to 1s
          // cmp_mask = _mm256_movemask_epi8(acc_le_heaptop); // if there is any pair that acc[i] == min(acc[i], heap_top) => new dist found
          
          
          __m256i flip = _mm256_set1_epi8(char(0x80));
          __m256i a = _mm256_xor_si256(acc, flip);
          __m256i b = _mm256_xor_si256(heap_top, flip);
          __m256i lt = _mm256_cmpgt_epi8(b, a);          // (b > a) signed  <=> acc < heap_top unsigned
          cmp_mask = _mm256_movemask_epi8(lt);

        } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
          assert(false && "NOT IMPLEMENTED!");
        }

        
        
        if(cmp_mask){
          alignas(32) uint8_t temp[32];
          _mm256_store_si256(reinterpret_cast<__m256i*>(temp), acc);

          while (cmp_mask) {
            int j = __builtin_ctz(cmp_mask); 
            res.push(temp[j], ID{idBias,  idmap[i + j]});
            cmp_mask &= cmp_mask - 1;            
          }

          heap_top = _mm256_set1_epi8(res.worst());
          //keyKernalSec++;

        }
        

    }



    const int start = codebookRow - (codebookRow % stepSize);
    const int end   = codebookRow;

    const int R    = curCodes.rows();     // list size
    const int M    = mSubspaceNum;

    // 1) cache each LUT column base pointer
    std::vector<const TargetDType*> lut_col(M);
    const TargetDType* lut_base = lut.data();
    for (int col = 0; col < M; ++col) {
        lut_col[col] = lut_base + col * ksub;
    }

    // 2) codes base pointer (ColMajor)
    const uint8_t* codes_base = curCodes.data();

    for (int i = start; i < end; ++i) {
        uint16_t acc16 = 0;

        // ColMajor: code(i,col) = codes_base[col*R + i]
        #pragma unroll
        for (int col = 0; col < numSubspaceNum; ++col) {
            const uint8_t code = codes_base[col * R + i];
            acc16 += (uint16_t)lut_col[col][code];
        }

        const uint8_t dist8 = (acc16 > 255u) ? 255u : (uint8_t)acc16;

        if (dist8 < res.worst()) {
            res.push(dist8, ID{idBias, idmap[i]});
        }
    }

    //f::heap_reorder<f::CMax<TargetDType, int>>(k, heap_dis, heap_ids);
}


template <typename TargetDType, int numSubspaceNum>
void PQ::searchHeapSIMD128Centroids(
  ColMatrix<TargetDType>& lut, const int k, int q_idx,
  TopKHeap& res, ColMatrix<uint8_t> &curCodes, std::vector<int>& idmap) {

    constexpr int numCentroid=128;

    const int ksub = lut.rows();    

    __m256i lut_registers[numSubspaceNum*4];
    for (int col = 0; col < 4*numSubspaceNum;) {
        lut_registers[col] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/4) * ksub));
        lut_registers[col + 1] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/4) * ksub + 32 ));
        lut_registers[col + 2] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/4) * ksub + 64 ));
        lut_registers[col + 3] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + (col/4) * ksub + 96 ));
        col += 4;
    }

    // curCodes is col-major
    uint8_t* codes = curCodes.data();


    int stepSize;
    const int codebookRow = curCodes.rows();

    __m256i heap_top;
    // whether there is a new dist less than heap_top
    int cmp_mask = 0;

    if constexpr (std::is_same<TargetDType, uint8_t>::value) {
      stepSize = 32;
      heap_top = _mm256_set1_epi8(res.worst());
    } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
      stepSize = 16;
      heap_top = _mm256_set1_epi16(0xFFFF);
    } else {
      assert(false);
    }


    const __m256i mask_0 = _mm256_set1_epi8(0x0F);
    const __m256i mask_1 = _mm256_set1_epi8(0x1F);
    const __m256i mask_2 = _mm256_set1_epi8(0x2F);
    const __m256i mask_3 = _mm256_set1_epi8(0x3F);
    const __m256i mask_4 = _mm256_set1_epi8(0x4F);
    const __m256i mask_5 = _mm256_set1_epi8(0x5F);
    const __m256i mask_6 = _mm256_set1_epi8(0x6F);


    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i mask_hi3 = _mm256_set1_epi8(0x07);
    const __m256i zero     = _mm256_setzero_si256();
    const __m256i ones     = _mm256_set1_epi8((char)0xFF);

    struct LUTPair{
      __m256i lo, hi;
    };

    alignas(32) LUTPair lutPairs[numSubspaceNum*4];    

    //auto start = std::chrono::steady_clock::now();
    for (int col = 0; col < numSubspaceNum*4; col += 4){
      lutPairs[col].lo = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x00);
      lutPairs[col].hi = _mm256_permute2x128_si256(lut_registers[col], lut_registers[col], 0x11);
      lutPairs[col + 1].lo = _mm256_permute2x128_si256(lut_registers[col + 1], lut_registers[col + 1], 0x00);
      lutPairs[col + 1].hi = _mm256_permute2x128_si256(lut_registers[col + 1], lut_registers[col + 1], 0x11);
      lutPairs[col + 2].lo = _mm256_permute2x128_si256(lut_registers[col + 2], lut_registers[col + 2], 0x00);
      lutPairs[col + 2].hi = _mm256_permute2x128_si256(lut_registers[col + 2], lut_registers[col + 2], 0x11);
      lutPairs[col + 3].lo = _mm256_permute2x128_si256(lut_registers[col + 3], lut_registers[col + 3], 0x00);
      lutPairs[col + 3].hi = _mm256_permute2x128_si256(lut_registers[col + 3], lut_registers[col + 3], 0x11);
    }

    // auto end = std::chrono::steady_clock::now();
    // keyKernalSec += std::chrono::duration<double>(end - start).count();


    for (int i = 0; i < codebookRow - codebookRow % stepSize; i += stepSize) {

        __m256i acc = _mm256_setzero_si256();

        for (int col = 0; col < numSubspaceNum*4;) {
            if constexpr (std::is_same<TargetDType, uint8_t>::value) {
                // 批量加载 32 个 codes
                if constexpr (numCentroid == 128) {
                  __m256i code_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + codebookRow*(col/4))); // 32 codes, 8 bits each




                  if (true) {

                    __m256i indices_low = _mm256_and_si256(code_vector, mask_0); // byte-wise AND, each 1 byte code only get its low 4 bits (0xXY -> 0x0Y)

                    __m256i result_0 = _mm256_shuffle_epi8(lutPairs[col].lo, indices_low); // look-up based on 0x0Y
                    __m256i result_1 = _mm256_shuffle_epi8(lutPairs[col].hi, indices_low); // look-up based on 0x1Y
                    
                    // select lut_0 if code is 0x0Y. select lut_1 if code is 0x1Y ~ 0x3Y
                    __m256i cmp = _mm256_cmpgt_epi8(code_vector, mask_0); 
                    __m256i result_01 = _mm256_blendv_epi8(result_0, result_1, cmp); // conditional selection

                    __m256i result_2 = _mm256_shuffle_epi8(lutPairs[col+1].lo, indices_low); // look-up based on 0x2Y
                    __m256i result_3 = _mm256_shuffle_epi8(lutPairs[col+1].hi, indices_low); // look-up based on 0x3Y
                    
                    // select lut_3 if code is 0x3Y. select lut_2 if code is 0x0Y ~ 0x2Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_2); 
                    __m256i result_23 = _mm256_blendv_epi8(result_2, result_3, cmp); // conditional selection

                    // select result_01 if code is 0x0Y or 0x1Y. select result_23 if code is 0x2Y or 0x3Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_1); 
                    __m256i result_0123 = _mm256_blendv_epi8(result_01, result_23, cmp);

                    __m256i result_4 = _mm256_shuffle_epi8(lutPairs[col+2].lo, indices_low); // look-up based on 0x0Y
                    __m256i result_5 = _mm256_shuffle_epi8(lutPairs[col+2].hi, indices_low); // look-up based on 0x1Y
                    
                    // select lut_4 if code is 0x0Y. select lut_5 if code is 0x1Y ~ 0x7Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_4); 
                    __m256i result_45 = _mm256_blendv_epi8(result_4, result_5, cmp); // conditional selection

                    __m256i result_6 = _mm256_shuffle_epi8(lutPairs[col+3].lo, indices_low); // look-up based on 0x2Y
                    __m256i result_7 = _mm256_shuffle_epi8(lutPairs[col+3].hi, indices_low); // look-up based on 0x3Y

                    cmp = _mm256_cmpgt_epi8(code_vector, mask_6); 
                    __m256i result_67 = _mm256_blendv_epi8(result_6, result_7, cmp); // conditional selection
                    
                    // select lut_3 if code is 0x3Y. select lut_2 if code is 0x0Y ~ 0x2Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_5); 
                    __m256i result_4567 = _mm256_blendv_epi8(result_45, result_67, cmp); // conditional selection

                    // select result_01 if code is 0x0Y or 0x1Y. select result_23 if code is 0x2Y or 0x3Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_3); 
                    __m256i dist = _mm256_blendv_epi8(result_0123, result_4567, cmp);

                    acc = _mm256_adds_epu8(acc, dist);
                    col += 4;
                  } else {
                    __m256i indices_low = _mm256_and_si256(code_vector, mask_0); // byte-wise AND, each 1 byte code only get its low 4 bits (0xXY -> 0x0Y)

                    __m256i result_0 = _mm256_shuffle_epi8(lutPairs[col].lo, indices_low); // look-up based on 0x0Y
                    __m256i result_1 = _mm256_shuffle_epi8(lutPairs[col].hi, indices_low); // look-up based on 0x1Y
                    
                    // select lut_0 if code is 0x0Y. select lut_1 if code is 0x1Y ~ 0x3Y
                    __m256i cmp = _mm256_cmpgt_epi8(code_vector, mask_0); 
                    __m256i result_01 = _mm256_blendv_epi8(result_0, result_1, cmp); // conditional selection

                    __m256i result_2 = _mm256_shuffle_epi8(lutPairs[col+1].lo, indices_low); // look-up based on 0x2Y
                    __m256i result_3 = _mm256_shuffle_epi8(lutPairs[col+1].hi, indices_low); // look-up based on 0x3Y
                    
                    // select lut_3 if code is 0x3Y. select lut_2 if code is 0x0Y ~ 0x2Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_2); 
                    __m256i result_23 = _mm256_blendv_epi8(result_2, result_3, cmp); // conditional selection

                    // select result_01 if code is 0x0Y or 0x1Y. select result_23 if code is 0x2Y or 0x3Y
                    cmp = _mm256_cmpgt_epi8(code_vector, mask_1); 
                    __m256i dist_vector = _mm256_blendv_epi8(result_01, result_23, cmp);

                    acc = _mm256_adds_epu8(acc, dist_vector);
                    col += 4;
                  }

                }
            } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
              // only AVX512 support _mm256_permute16x16_si256
              // so we need to use _mm256_permute8x32_si256 instead for AVX2, then why not use uint32
              assert(false && "NOT IMPLEMENTED!");
            }


        }

        codes += stepSize;
        __m256i cmp_result;
        if constexpr (std::is_same<TargetDType, uint8_t>::value) {
          // cmp_result = _mm256_cmpgt_epi8(heap_top, acc); // avx2 not supoort uint8 cmp, only int8 cmp

          // __m256i min_u = _mm256_min_epu8(acc, heap_top);
          // __m256i acc_le_heaptop = _mm256_cmpeq_epi8(min_u, acc); // if a pair is equal, set to 1s
          // cmp_mask = _mm256_movemask_epi8(acc_le_heaptop); // if there is any pair that acc[i] == min(acc[i], heap_top) => new dist found
          
          
          __m256i flip = _mm256_set1_epi8(char(0x80));
          __m256i a = _mm256_xor_si256(acc, flip);
          __m256i b = _mm256_xor_si256(heap_top, flip);
          __m256i lt = _mm256_cmpgt_epi8(b, a);          // (b > a) signed  <=> acc < heap_top unsigned
          cmp_mask = _mm256_movemask_epi8(lt);

        } else if constexpr (std::is_same<TargetDType, uint16_t>::value) {
          assert(false && "NOT IMPLEMENTED!");
        }

        
        
        if(cmp_mask){
          alignas(32) uint8_t temp[32];
          _mm256_store_si256(reinterpret_cast<__m256i*>(temp), acc);

          while (cmp_mask) {
            int j = __builtin_ctz(cmp_mask); 
            res.push(temp[j], ID{idBias,  idmap[i + j]});
            cmp_mask &= cmp_mask - 1;           
          }

          heap_top = _mm256_set1_epi8(res.worst());
          //keyKernalSec++;

        }
        

    }


    const int start = codebookRow - (codebookRow % stepSize);
    const int end   = codebookRow;

    const int R    = curCodes.rows();     // list size
    const int M    = mSubspaceNum;

    // 1) cache each LUT column base pointer
    std::vector<const TargetDType*> lut_col(M);
    const TargetDType* lut_base = lut.data();
    for (int col = 0; col < M; ++col) {
        lut_col[col] = lut_base + col * ksub;
    }

    // 2) codes base pointer (ColMajor)
    const uint8_t* codes_base = curCodes.data();

    for (int i = start; i < end; ++i) {
        uint16_t acc16 = 0;

        // ColMajor: code(i,col) = codes_base[col*R + i]
        #pragma unroll
        for (int col = 0; col < numSubspaceNum; ++col) {
            const uint8_t code = codes_base[col * R + i];
            acc16 += (uint16_t)lut_col[col][code];
        }

        const uint8_t dist8 = (acc16 > 255u) ? 255u : (uint8_t)acc16;

        if (dist8 < res.worst()) {
            res.push(dist8, ID{idBias, idmap[i]});
        }
    }


    //f::heap_reorder<f::CMax<TargetDType, int>>(k, heap_dis, heap_ids);
}



#if defined(__AVX512VBMI__) && defined(__AVX512BW__) && defined(__AVX512F__)

template<int numSubspaceNum>
void PQ::searchHeapSIMD256Centroids(
    ColMatrix<uint8_t>& lut,                 // (ksub=256) x (M=numSubspaceNum), col-major
    const int k,
    TopKHeap& res,
    ColMatrix<uint8_t>& curCodes,            // (R x M), col-major, each entry in [0,255]
    std::vector<int>& idmap                  // size R, local->original
) {
    constexpr int numCentroid = 256;
    constexpr int batchSize   = 64;

    (void)k;

    const int ksub = lut.rows();           // should be 256
    const int codebookRow = curCodes.rows();

    assert(ksub == numCentroid);
    assert(curCodes.cols() == numSubspaceNum);

    // curCodes is col-major: codes layout = [col0 all rows][col1 all rows]...
    uint8_t* codes = curCodes.data();

    // StepSize == batchSize for u8 x 64 lanes
    const int stepSize = batchSize;

    // Heap top broadcast
    __m512i heap_top = _mm512_set1_epi8((char)res.worst());

    // constants
    const __m512i flip80 = _mm512_set1_epi8((char)0x80);
    const __m512i mask7f = _mm512_set1_epi8((char)0x7F);

    // Step 1: preload LUT chunks: for each subspace, 4 x 64-byte blocks
    // lut is col-major: lut.data() points to column 0 first (256 bytes), then column 1, etc.
    __m512i lut_registers[4 * numSubspaceNum];
    for (int col = 0; col < numSubspaceNum; ++col) {
        const uint8_t* base = lut.data() + col * ksub; // start of this subspace's 256-entry LUT
        lut_registers[col * 4 + 0] = _mm512_loadu_si512((const void*)(base +  0));  // 0..63
        lut_registers[col * 4 + 1] = _mm512_loadu_si512((const void*)(base + 64));  // 64..127
        lut_registers[col * 4 + 2] = _mm512_loadu_si512((const void*)(base + 128)); // 128..191
        lut_registers[col * 4 + 3] = _mm512_loadu_si512((const void*)(base + 192)); // 192..255
    }

    alignas(64) uint8_t temp[batchSize];

    // Main loop: 64 vectors each iteration
    int i = 0;
    for (; i + stepSize <= codebookRow; i += stepSize) {
        __m512i acc = _mm512_setzero_si512();

        // accumulate over subspaces
        for (int col = 0; col < numSubspaceNum; ++col) {
            // load 64 codes for this subspace at rows [i, i+63]
            const uint8_t* cptr = curCodes.data() + col * codebookRow + i;
            __m512i code_vector = _mm512_loadu_si512((const void*)cptr);

            // select high half (128..255) vs low half (0..127)
            __mmask64 sel_mask = _mm512_test_epi8_mask(code_vector, flip80); // test bit7

            // idx in [0..127]
            __m512i idx = _mm512_and_si512(code_vector, mask7f);

            // LUT for 0..127:
            __m512i result_01 = _mm512_permutex2var_epi8(
                lut_registers[col * 4 + 0], idx, lut_registers[col * 4 + 1]);

            // LUT for 128..255 (mapped by same idx 0..127):
            __m512i result_23 = _mm512_permutex2var_epi8(
                lut_registers[col * 4 + 2], idx, lut_registers[col * 4 + 3]);

            // pick based on bit7 of original code
            __m512i dist = _mm512_mask_blend_epi8(sel_mask, result_01, result_23);

            // saturating add u8
            acc = _mm512_adds_epu8(acc, dist);
        }

        // Compare acc < heap_top (unsigned u8):
        // use sign-flip trick: unsigned compare == signed compare after xor 0x80
        __m512i a = _mm512_xor_si512(acc, flip80);
        __m512i b = _mm512_xor_si512(heap_top, flip80);
        __mmask64 m = _mm512_cmpgt_epi8_mask(b, a); // (b > a) signed  <=> acc < heap_top unsigned

        if (m) {
            _mm512_store_si512((void*)temp, acc);

            uint64_t mm = (uint64_t)m;
            while (mm) {
                int j = (int)__builtin_ctzll(mm);
                res.push(temp[j], ID{ idBias, idmap[i + j] });
                mm &= (mm - 1);
            }

            heap_top = _mm512_set1_epi8((char)res.worst());
        }
    }

    // Tail (scalar)
    for (; i < codebookRow; ++i) {
        uint8_t dist = 0;
        for (int col = 0; col < numSubspaceNum; ++col) {
            const uint8_t code = curCodes(i, col);          // NOTE: your ColMatrix may need operator()(row,col)
            dist = unsigendSaturatedAdd<uint8_t>(dist, lut(code, col)); // same note
        }
        if (dist < res.worst()) {
            res.push(dist, ID{ idBias, idmap[i] });
        }
    }
}

// Helper: scalar saturated add for uint8_t
static inline uint8_t u8_sat_add(uint8_t a, uint8_t b) {
    unsigned s = (unsigned)a + (unsigned)b;
    return (s > 255u) ? 255u : (uint8_t)s;
}

// Helper: get bit8 (packed) for a scalar row i in a given subspace col
static inline uint8_t get_bit8_packed(const uint64_t* packed, int nBlocks, int col, int row) {
    const int blk = row >> 6;          // /64
    const int off = row & 63;          // %64
    const uint64_t w = packed[col * nBlocks + blk];
    return (uint8_t)((w >> off) & 1ull);
}

template<int numSubspaceNum>
void PQ::searchHeapSIMD512Centroids(
    ColMatrix<uint8_t>& lut,                 // (ksub=512) x (M=numSubspaceNum), col-major
    const int k,
    TopKHeap& res,
    ColMatrix<uint8_t>& curCodes8,           // (R x M), col-major, low 8 bits of code
    ColMatrix<uint16_t>& tailCodes,
    const uint64_t *codeHiPacked,            // packed bit8, layout: [col0 blocks][col1 blocks]...
    std::vector<int>& idmap                  // size R, local->original
) {
    constexpr int numCentroid = 512;
    constexpr int batchSize   = 64;

    (void)k; // keep signature consistent with your 8-bit version

    const int ksub = lut.rows();           // should be 512
    const int codebookRow = curCodes8.rows();
    assert(ksub == numCentroid);
    assert(curCodes8.cols() == numSubspaceNum);

    // number of 64-row blocks for packed bit8
    const int nBlocks = (codebookRow + 63) / 64;

    // heap top broadcast
    __m512i heap_top = _mm512_set1_epi8((char)res.worst());

    // constants
    const __m512i flip80 = _mm512_set1_epi8((char)0x80);
    const __m512i mask7f = _mm512_set1_epi8((char)0x7F);

    // Step 1: preload LUT chunks: for each subspace, 8 x 64-byte blocks (512 entries)
    // lut is col-major: each column is 512 bytes
    __m512i lut_registers[8 * numSubspaceNum];
    for (int col = 0; col < numSubspaceNum; ++col) {
        const uint8_t* base = lut.data() + col * ksub; // start of this subspace's 512-entry LUT
        lut_registers[col * 8 + 0] = _mm512_loadu_si512((const void*)(base +   0));  // 0..63
        lut_registers[col * 8 + 1] = _mm512_loadu_si512((const void*)(base +  64));  // 64..127
        lut_registers[col * 8 + 2] = _mm512_loadu_si512((const void*)(base + 128));  // 128..191
        lut_registers[col * 8 + 3] = _mm512_loadu_si512((const void*)(base + 192));  // 192..255
        lut_registers[col * 8 + 4] = _mm512_loadu_si512((const void*)(base + 256));  // 256..319
        lut_registers[col * 8 + 5] = _mm512_loadu_si512((const void*)(base + 320));  // 320..383
        lut_registers[col * 8 + 6] = _mm512_loadu_si512((const void*)(base + 384));  // 384..447
        lut_registers[col * 8 + 7] = _mm512_loadu_si512((const void*)(base + 448));  // 448..511
    }

    alignas(64) uint8_t temp[batchSize];

    // Main loop: 64 vectors each iteration
    int i = 0;
    for (; i + batchSize <= codebookRow; i += batchSize) {
        __m512i acc = _mm512_setzero_si512();

        // block index for packed bit8
        const int blk = i >> 6; // i/64

        __mmask64 b8mask[numSubspaceNum];
        #pragma unroll
        for (int col = 0; col < numSubspaceNum; ++col)
            b8mask[col] = (__mmask64)codeHiPacked[col * nBlocks + blk];

        // accumulate over subspaces
        #pragma unroll
        for (int col = 0; col < numSubspaceNum; ++col) {
            _mm_prefetch((const char*)(curCodes8.data() + col * codebookRow + i + 64), _MM_HINT_T0);
            // load 64 codes (low 8 bits) for this subspace at rows [i, i+63]
            const uint8_t* cptr = curCodes8.data() + col * codebookRow + i;
            __m512i code_vector = _mm512_loadu_si512((const void*)cptr);

            // b7: bit7 of low byte (distinguish 0..127 vs 128..255 within a 256-half)
            __mmask64 b7 = _mm512_test_epi8_mask(code_vector, flip80);

            // idx in [0..127] = low 7 bits
            __m512i idx = _mm512_and_si512(code_vector, mask7f);

            // b8: packed 9th bit for these 64 rows in this subspace
            // bit j corresponds to row (i + j)
            //__mmask64 b8 = (__mmask64)codeHiPacked[col * nBlocks + blk];
             __mmask64 b8 = b8mask[col];

            // 4 groups (each is 128-entry LUT addressed by idx):
            // group0: b8=0,b7=0 -> codes 0..127   -> LUT[0..127]   = blocks 0,1
            // group1: b8=0,b7=1 -> codes 128..255 -> LUT[128..255] = blocks 2,3
            // group2: b8=1,b7=0 -> codes 256..383 -> LUT[256..383] = blocks 4,5
            // group3: b8=1,b7=1 -> codes 384..511 -> LUT[384..511] = blocks 6,7
            __m512i r0 = _mm512_permutex2var_epi8(
                lut_registers[col * 8 + 0], idx, lut_registers[col * 8 + 1]);
            __m512i r1 = _mm512_permutex2var_epi8(
                lut_registers[col * 8 + 2], idx, lut_registers[col * 8 + 3]);
            __m512i r2 = _mm512_permutex2var_epi8(
                lut_registers[col * 8 + 4], idx, lut_registers[col * 8 + 5]);
            __m512i r3 = _mm512_permutex2var_epi8(
                lut_registers[col * 8 + 6], idx, lut_registers[col * 8 + 7]);

            // build masks for 4-way select
            // m0 = (~b8)&(~b7) (implicit by starting from r0)
            // 2-level binary selection: (b8 -> b7)
            __m512i p0 = _mm512_mask_blend_epi8(b8, r0, r2); // b8=0 -> r0, b8=1 -> r2
            __m512i p1 = _mm512_mask_blend_epi8(b8, r1, r3); // b8=0 -> r1, b8=1 -> r3
            __m512i dist = _mm512_mask_blend_epi8(b7, p0, p1); // b7=0 -> p0, b7=1 -> p1


            // saturating add u8
            acc = _mm512_adds_epu8(acc, dist);
        }

        // Compare acc < heap_top (unsigned u8) via sign-flip trick
        __m512i a = _mm512_xor_si512(acc, flip80);
        __m512i b = _mm512_xor_si512(heap_top, flip80);
        __mmask64 m = _mm512_cmpgt_epi8_mask(b, a); // (b > a) signed  <=> acc < heap_top unsigned

        if (m) {
            _mm512_store_si512((void*)temp, acc);

            uint64_t mm = (uint64_t)m;
            while (mm) {
                int j = (int)__builtin_ctzll(mm);
                res.push(temp[j], ID{ idBias, idmap[i + j] });
                mm &= (mm - 1);
            }

            heap_top = _mm512_set1_epi8((char)res.worst());
        }
    }

    // Tail (scalar)
    const int tailNum = tailCodes.rows();
    const int tailStart = codebookRow & ~63;
    for(int i = 0; i < tailNum; ++i) {
        uint16_t acc = 0;
        for (int col = 0; col < numSubspaceNum; ++col) {
            const uint16_t code = tailCodes(i, col);
            acc += lut(code, col);
        }
        const uint8_t dist = (acc > 255) ? 255 : (uint8_t)acc;
        if (dist < res.worst()) {
            res.push(dist, ID{ idBias, idmap[i+tailStart] });
        }
    }
}

static inline uint8_t get_bit_packed(const uint64_t* packed,
                                     const int nBlocks,
                                     const int col,
                                     const int row) {
    const int blk = row >> 6;     // row / 64
    const int off = row & 63;     // row % 64
    const uint64_t w = packed[(size_t)col * (size_t)nBlocks + (size_t)blk];
    return (uint8_t)((w >> off) & 1ULL);
}

template<int numSubspaceNum>
void PQ::searchHeapSIMD1024Centroids(
    ColMatrix<uint8_t>& lut,               // (ksub=1024) x M, col-major
    const int k,
    TopKHeap& res,
    ColMatrix<uint8_t>& curCodes8,         // (R x M), col-major, low 8 bits
    ColMatrix<uint16_t>& tailCodes,
    const uint64_t* codeHiPacked,        // packed bit8: [col0 blocks][col1 blocks]...
    const uint64_t* codeHiPacked9,        // packed bit9: [col0 blocks][col1 blocks]...
    std::vector<int>& idmap
) {
    constexpr int numCentroid = 1024;
    constexpr int batchSize   = 64;

    (void)k;

    const int ksub = lut.rows();           // should be 1024
    const int R    = curCodes8.rows();
    assert(ksub == numCentroid);
    assert(curCodes8.cols() == numSubspaceNum);

    const int nBlocks = (R + 63) / 64;

    __m512i heap_top = _mm512_set1_epi8((char)res.worst());

    const __m512i flip80 = _mm512_set1_epi8((char)0x80);
    const __m512i mask7f = _mm512_set1_epi8((char)0x7F);

    // 1024-entry LUT => 16 blocks of 64B per subspace
    __m512i lut_registers[16 * numSubspaceNum];
    for (int col = 0; col < numSubspaceNum; ++col) {
        const uint8_t* base = lut.data() + col * ksub;
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            lut_registers[col * 16 + t] =
                _mm512_loadu_si512((const void*)(base + t * 64));
        }
    }

    alignas(64) uint8_t temp[batchSize];

    int i = 0;
    for (; i + batchSize <= R; i += batchSize) {
        __m512i acc = _mm512_setzero_si512();
        const int blk = i >> 6; // /64

        for (int col = 0; col < numSubspaceNum; ++col) {
            const uint8_t* cptr = curCodes8.data() + col * R + i;
            __m512i code_vector = _mm512_loadu_si512((const void*)cptr);

            // b7 from low byte
            __mmask64 b7 = _mm512_test_epi8_mask(code_vector, flip80);

            // idx = low 7 bits
            __m512i idx = _mm512_and_si512(code_vector, mask7f);

            // b8, b9 from packed
            __mmask64 b8 = (__mmask64)codeHiPacked[col * nBlocks + blk];
            __mmask64 b9 = (__mmask64)codeHiPacked9[col * nBlocks + blk];

            // 8 groups, each group covers 128 entries => 2 LUT blocks
            __m512i r0 = _mm512_permutex2var_epi8(
                lut_registers[col * 16 + 0], idx, lut_registers[col * 16 + 1]);
            __m512i r1 = _mm512_permutex2var_epi8(
                lut_registers[col * 16 + 2], idx, lut_registers[col * 16 + 3]);
            __m512i r2 = _mm512_permutex2var_epi8(
                lut_registers[col * 16 + 4], idx, lut_registers[col * 16 + 5]);
            __m512i r3 = _mm512_permutex2var_epi8(
                lut_registers[col * 16 + 6], idx, lut_registers[col * 16 + 7]);
            __m512i r4 = _mm512_permutex2var_epi8(
                lut_registers[col * 16 + 8], idx, lut_registers[col * 16 + 9]);
            __m512i r5 = _mm512_permutex2var_epi8(
                lut_registers[col * 16 +10], idx, lut_registers[col * 16 +11]);
            __m512i r6 = _mm512_permutex2var_epi8(
                lut_registers[col * 16 +12], idx, lut_registers[col * 16 +13]);
            __m512i r7 = _mm512_permutex2var_epi8(
                lut_registers[col * 16 +14], idx, lut_registers[col * 16 +15]);

            // step 1: within b9=0 half (r0,r1,r2,r3)
            // b8 selects (r0 vs r2) and (r1 vs r3)
            __m512i lo_p0 = _mm512_mask_blend_epi8(b8, r0, r2); // b8=0 -> r0, b8=1 -> r2
            __m512i lo_p1 = _mm512_mask_blend_epi8(b8, r1, r3); // b8=0 -> r1, b8=1 -> r3
            // b7 selects between the two pairs
            __m512i lo9   = _mm512_mask_blend_epi8(b7, lo_p0, lo_p1); // b7=0 -> lo_p0, b7=1 -> lo_p1

            // step 2: within b9=1 half (r4,r5,r6,r7)
            __m512i hi_p0 = _mm512_mask_blend_epi8(b8, r4, r6); // b8=0 -> r4, b8=1 -> r6
            __m512i hi_p1 = _mm512_mask_blend_epi8(b8, r5, r7); // b8=0 -> r5, b8=1 -> r7
            __m512i hi9   = _mm512_mask_blend_epi8(b7, hi_p0, hi_p1); // b7=0 -> hi_p0, b7=1 -> hi_p1

            // step 3: b9 selects low-half vs high-half
            __m512i dist  = _mm512_mask_blend_epi8(b9, lo9, hi9); // b9=0 -> lo9, b9=1 -> hi9

            acc = _mm512_adds_epu8(acc, dist);
        }

        // unsigned acc < heap_top via sign-flip
        __m512i a = _mm512_xor_si512(acc, flip80);
        __m512i b = _mm512_xor_si512(heap_top, flip80);
        __mmask64 m = _mm512_cmpgt_epi8_mask(b, a);

        if (m) {
            _mm512_store_si512((void*)temp, acc);
            uint64_t mm = (uint64_t)m;
            while (mm) {
                int j = (int)__builtin_ctzll(mm);
                res.push(temp[j], ID{ idBias, idmap[i + j] });
                mm &= (mm - 1);
            }
            heap_top = _mm512_set1_epi8((char)res.worst());
        }
    }

    // tail scalar
    const int tailNum = tailCodes.rows();
    const int tailStart = R & ~63;
    for(int i = 0; i < tailNum; ++i) {
        uint16_t acc = 0;
        for (int col = 0; col < numSubspaceNum; ++col) {
            const uint16_t code = tailCodes(i, col);
            acc += lut(code, col);
        }
        const uint8_t dist = (acc > 255) ? 255 : (uint8_t)acc;
        if (dist < res.worst()) {
            res.push(dist, ID{ idBias, idmap[i+tailStart] });
        }
    }
}


#else
template<int numSubspaceNum>
void PQ::searchHeapSIMD256Centroids(
    ColMatrix<uint8_t>& lut,                 // (ksub=256) x (M=numSubspaceNum), col-major
    const int k,
    TopKHeap& res,
    ColMatrix<uint8_t>& curCodes,            // (R x M), col-major, each entry in [0,255]
    std::vector<int>& idmap                  // size R, local->original
) {
    static_assert(numSubspaceNum == -1,
        "searchHeapSIMD256Centroids needs AVX-512 VBMI (+BW,+F). Compile with -mavx512vbmi -mavx512bw -mavx512f");
}

template<int numSubspaceNum>
void PQ::searchHeapSIMD512Centroids(
    ColMatrix<uint8_t>& lut,                 // (ksub=512) x (M=numSubspaceNum), col-major
    const int k,
    TopKHeap& res,
    ColMatrix<uint8_t>& curCodes8,           // (R x M), col-major, low 8 bits of code
    ColMatrix<uint16_t>& tailCodes,
    const uint64_t *codeHiPacked,            // packed bit8, layout: [col0 blocks][col1 blocks]...
    std::vector<int>& idmap                  // size R, local->original
) {
    static_assert(numSubspaceNum == -1,
        "searchHeapSIMD512Centroids needs AVX-512 VBMI (+BW,+F). Compile with -mavx512vbmi -mavx512bw -mavx512f");
}



template<int numSubspaceNum>
void PQ::searchHeapSIMD1024Centroids(
    ColMatrix<uint8_t>& lut,               // (ksub=1024) x M, col-major
    const int k,
    TopKHeap& res,
    ColMatrix<uint8_t>& curCodes8,         // (R x M), col-major, low 8 bits
    ColMatrix<uint16_t>& tailCodes,
    const uint64_t* codeHiPacked,        // packed bit8: [col0 blocks][col1 blocks]...
    const uint64_t* codeHiPacked9,        // packed bit9: [col0 blocks][col1 blocks]...
    std::vector<int>& idmap
) {
    static_assert(numSubspaceNum == -1,
        "searchHeapSIMD1024Centroids needs AVX-512 VBMI (+BW,+F). Compile with -mavx512vbmi -mavx512bw -mavx512f");
}


#endif


#endif  // PQ_H