#include <algorithm>
#include <utility>
#include <armadillo>
#include <execution>

#include "PQ.h"
#include "utils/TimingUtils.hpp"
#include "utils/IO.hpp"

thread_local LUTType PQ::lut;

void PQ::normalize(RowMatrixXf &data) {
  RowVectorXf mean = data.colwise().mean();
  RowVectorXf stdDev = ((data.rowwise() - mean).array().square().colwise().sum() / (data.rows() - 1)).sqrt();
  data.rowwise() -= mean;
  data.array().rowwise() /= stdDev.array();
} 

void PQ::train(RowMatrixXf &XTrain, bool verbose) {

  if (mCentroidsNum > XTrain.rows()) {
        std::cout << "#Centroids <= #Sample, it's impossible for KMeans. " << mCentroidsNum << " should <= " << XTrain.rows() << std::endl;
        assert(false);
  }

  codebookRow = XTrain.rows();

  const int oriTotalDim = XTrain.cols();
  mSubsLen = oriTotalDim / mSubspaceNum;
  if (oriTotalDim % mSubspaceNum > 0) {
    mSubsLen += 1;
  }

  // learn dictionary
  if (mCentroidsPerSubs.size() == 0) {
    mCentroidsPerSubs.resize(mSubspaceNum);
    for (int iSubs=0; iSubs<mSubspaceNum; iSubs++) {

        // int sampleSize = std::min(mCentroidsNum * 256, (int)XTrain.rows());
        const int sampleSize = std::min(static_cast<int>(XTrain.rows()), int(1e7));
        RowMatrixXf XTrainSlice(sampleSize, mSubsLen);
        std::vector<int> perm(XTrain.rows());
        randomPermutation(perm);
        for (int i=0; i<sampleSize; i++) {
          XTrainSlice.row(i).noalias() = XTrain.block(perm[i], iSubs * mSubsLen, 1, mSubsLen);
        }
    
        omp_set_num_threads(64);
        if (verbose) 
            std::cout << "subspace " << iSubs << " regular kmeans" << std::endl;
        mCentroidsPerSubs[iSubs].resize(mCentroidsNum, mSubsLen);
        arma::fmat means(mCentroidsPerSubs[iSubs].data(), mSubsLen, mCentroidsNum, false, false);

        arma::fmat data(XTrainSlice.data(), XTrainSlice.cols(), XTrainSlice.rows(), false, false);

        bool status = arma::kmeans(means, data, mCentroidsNum, arma::static_subset, 50, false);
        if (status == false) {
            std::cout << "kmeans arma failed" << std::endl;
            exit(0);
        }
 
    }
  }
  // create centroids col major version to enable fast LUT creation
// #ifdef __AVX2__
//   mCentroidsPerSubsCMajor.resize(mCentroidsPerSubs.size());
//   for (int i=0; i<(int)mCentroidsPerSubs.size(); i++) {
//     mCentroidsPerSubsCMajor[i] = mCentroidsPerSubs[i];
//   }
// #endif
}

static inline int assignCoarse1(const float* x, const float* coarse, int nlist, int dim) {
  int best = 0;
  float bestDist = std::numeric_limits<float>::infinity();
  for (int c = 0; c < nlist; ++c) {
    const float* cc = coarse + c * dim;
    float dist = 0.f;
    for (int d = 0; d < dim; ++d) {
      float diff = x[d] - cc[d];
      dist += diff * diff;
    }
    if (dist < bestDist) { bestDist = dist; best = c; }
  }
  return best;
}


// void PQ::trainIVF(RowMatrixXf &XTrain, const int nlist, bool verbose) {
//   this->mIVF.mIVFLists = nlist;

//   const int N   = static_cast<int>(XTrain.rows());
//   const int dim = static_cast<int>(XTrain.cols());

//   if (mCentroidsNum > N) {
//     std::cout << "#Centroids <= #Sample, impossible for KMeans. "
//               << mCentroidsNum << " should <= " << N << "\n";
//     assert(false);
//   }
//   if (mIVF.mIVFLists <= 0) {
//     std::cout << "Invalid mIVFLists: " << mIVF.mIVFLists << " (N=" << N << ")\n";
//     assert(false);
//   }

//   if(N / mIVF.mIVFLists <= 64) {
//     mIVF.mIVFLists = std::max(1, N / 64);
//   }

//   codebookRow = N;

//   // ------------------------------------------------------------
//   // 0) Assume XTrain is already padded so that dim is divisible by M
//   // ------------------------------------------------------------
//   if (dim % mSubspaceNum != 0) {
//     std::cout << "[trainIVF] ERROR: XTrain.cols()=" << dim
//               << " is not divisible by mSubspaceNum=" << mSubspaceNum
//               << ". If you said you padded, please pad to (M * subsLen).\n";
//     assert(false);
//   }
//   mSubsLen = dim / mSubspaceNum;

//   // A small helper: make a random permutation [0..N-1]
//   auto makePerm = [&](int n) {
//     std::vector<int> perm(n);
//     for (int i = 0; i < n; ++i) perm[i] = i;
//     // you likely already have randomPermutation(perm); use yours if you want
//     std::mt19937 rng(12345); // or use your RNG
//     std::shuffle(perm.begin(), perm.end(), rng);
//     return perm;
//   };

//   // ------------------------------------------------------------
//   // (A) Train IVF coarse centroids (global)
//   //   - data is (dim x N), each column is one sample vector
//   // ------------------------------------------------------------
//   if (mIVF.nlist == 0 || mIVF.coarseCentroids.rows() == 0) {
//     mIVF.nlist = mIVF.mIVFLists;
//     mIVF.coarseCentroids.resize(mIVF.mIVFLists, dim);

//     if (verbose) {
//       std::cout << "[IVF] coarse kmeans: nlist=" << mIVF.mIVFLists
//                 << " dim=" << dim << " N=" << N << "\n";
//     }

//     // Build arma data: dim x N, col i = XTrain.row(i)^T
//     arma::fmat data(dim, N);
//     for (int i = 0; i < N; ++i) {
//       const float* src = XTrain.row(i).data(); // contiguous (RowMajor row)
//       float* dst = data.colptr(i);
//       // copy dim floats
//       std::memcpy(dst, src, sizeof(float) * dim);
//     }

//     // arma means: dim x nlist (each column is a centroid)
//     arma::fmat means(dim, mIVF.mIVFLists);

//     omp_set_num_threads(64);
//     bool ok = arma::kmeans(means, data, mIVF.mIVFLists, arma::static_subset, 50, false);
//     if (!ok) {
//       std::cout << "coarse kmeans arma failed\n";
//       assert(false);
//     }

//     // Copy back to Eigen: mIVF.coarseCentroids is (nlist x dim), row c = centroid c
//     for (int c = 0; c < mIVF.mIVFLists; ++c) {
//       const float* src = means.colptr(c); // length dim
//       float* dst = mIVF.coarseCentroids.row(c).data();
//       std::memcpy(dst, src, sizeof(float) * dim);
//     }
//   }

//   // init lists (filled later in add())
//   mIVF.lists.clear();
//   mIVF.lists.resize(mIVF.nlist);

//   // ------------------------------------------------------------
//   // (B) Train global PQ codebook (shared by all lists)
//   //   For each subspace s:
//   //     - data is (mSubsLen x sampleSize), each col is one sample sub-vector
//   //     - means is (mSubsLen x K)
//   // ------------------------------------------------------------
//   if (mCentroidsPerSubs.empty()) {
//     mCentroidsPerSubs.resize(mSubspaceNum);

//     for (int s = 0; s < mSubspaceNum; ++s) {
//       const int sampleSize = std::min(N, int(1e7)); // consider reducing if memory heavy
//       const int startCol = s * mSubsLen;

//       if (verbose) {
//         std::cout << "[PQ] subspace " << s
//                   << " kmeans (sampleSize=" << sampleSize
//                   << ", dsub=" << mSubsLen << ")\n";
//       }

//       // pick sample indices
//       std::vector<int> perm = makePerm(N);

//       // Build arma data: dsub x sampleSize
//       arma::fmat data(mSubsLen, sampleSize);
//       for (int i = 0; i < sampleSize; ++i) {
//         const int rid = perm[i];
//         const float* src = XTrain.row(rid).data() + startCol; // subvector start
//         float* dst = data.colptr(i);
//         std::memcpy(dst, src, sizeof(float) * mSubsLen);
//       }

//       // arma means: dsub x K
//       arma::fmat means(mSubsLen, mCentroidsNum);

//       omp_set_num_threads(64);
//       bool ok = arma::kmeans(means, data, mCentroidsNum, arma::static_subset, 50, false);
//       if (!ok) {
//         std::cout << "pq kmeans arma failed\n";
//         assert(false);
//       }

//       // Copy back to Eigen: want (K x dsub), row k = centroid k
//       mCentroidsPerSubs[s].resize(mCentroidsNum, mSubsLen);
//       for (int k = 0; k < mCentroidsNum; ++k) {
//         const float* src = means.colptr(k); // length dsub
//         float* dst = mCentroidsPerSubs[s].row(k).data();
//         std::memcpy(dst, src, sizeof(float) * mSubsLen);
//       }
//     }
//   }

//   mIVF.coarseNorm2 = mIVF.coarseCentroids.rowwise().squaredNorm();

// }


// void PQ::trainIVF(RowMatrixXf &XTrain, const int nlist, bool verbose) {
//   this->mIVF.mIVFLists = nlist;

//   const int N   = static_cast<int>(XTrain.rows());
//   const int dim = static_cast<int>(XTrain.cols());
//   omp_set_num_threads(64);

//   if (mCentroidsNum > N) {
//     std::cout << "#Centroids <= #Sample, impossible for KMeans. "
//               << mCentroidsNum << " should <= " << N << "\n";
//     assert(false);
//   }
//   if (mIVF.mIVFLists <= 0) {
//     std::cout << "Invalid mIVFLists: " << mIVF.mIVFLists << " (N=" << N << ")\n";
//     assert(false);
//   }

//   if(N / mIVF.mIVFLists <= 64) {
//     mIVF.mIVFLists = std::max(1, N / 64);
//   }

//   codebookRow = N;

//   // ------------------------------------------------------------
//   // 0) Assume XTrain is already padded so that dim is divisible by M
//   // ------------------------------------------------------------
//   if (dim % mSubspaceNum != 0) {
//     std::cout << "[trainIVF] ERROR: XTrain.cols()=" << dim
//               << " is not divisible by mSubspaceNum=" << mSubspaceNum
//               << ". If you said you padded, please pad to (M * subsLen).\n";
//     assert(false);
//   }
//   mSubsLen = dim / mSubspaceNum;

//   // A small helper: make a random permutation [0..N-1]
//   auto makePerm = [&](int n) {
//     std::vector<int> perm(n);
//     for (int i = 0; i < n; ++i) perm[i] = i;
//     // you likely already have randomPermutation(perm); use yours if you want
//     std::mt19937 rng(12345); // or use your RNG
//     std::shuffle(perm.begin(), perm.end(), rng);
//     return perm;
//   };

//   // ------------------------------------------------------------
//   // (A) Train IVF coarse centroids (global)
//   //   - data is (dim x N), each column is one sample vector
//   // ------------------------------------------------------------
//   if (mIVF.nlist == 0 || mIVF.coarseCentroids.rows() == 0) {
//     mIVF.nlist = mIVF.mIVFLists;
//     mIVF.coarseCentroids.resize(mIVF.mIVFLists, dim);

//     if (verbose) {
//       std::cout << "[IVF] coarse kmeans: nlist=" << mIVF.mIVFLists
//                 << " dim=" << dim << " N=" << N << "\n";
//     }

//     // Build arma data: dim x N, col i = XTrain.row(i)^T
//     arma::fmat data(dim, N);
//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < N; ++i) {
//       const float* src = XTrain.row(i).data(); // contiguous (RowMajor row)
//       float* dst = data.colptr(i);
//       // copy dim floats
//       std::memcpy(dst, src, sizeof(float) * dim);
//     }

//     // arma means: dim x nlist (each column is a centroid)
//     arma::fmat means(dim, mIVF.mIVFLists);

//     bool ok = arma::kmeans(means, data, mIVF.mIVFLists, arma::static_subset, 50, false);
//     if (!ok) {
//       std::cout << "coarse kmeans arma failed\n";
//       assert(false);
//     }

//     puts("Finish kmeans");

//     // Copy back to Eigen: mIVF.coarseCentroids is (nlist x dim), row c = centroid c
//     #pragma omp parallel for schedule(static)
//     for (int c = 0; c < mIVF.mIVFLists; ++c) {
//       const float* src = means.colptr(c); // length dim
//       float* dst = mIVF.coarseCentroids.row(c).data();
//       std::memcpy(dst, src, sizeof(float) * dim);
//     }

//     puts("Finish writeback");

//   }

//   // init lists (filled later in add())
//   mIVF.lists.clear();
//   mIVF.lists.resize(mIVF.nlist);

//   // ------------------------------------------------------------
//   // (B) Train global PQ codebook (shared by all lists)
//   //   For each subspace s:
//   //     - data is (mSubsLen x sampleSize), each col is one sample sub-vector
//   //     - means is (mSubsLen x K)
//   // ------------------------------------------------------------
//   if (mCentroidsPerSubs.empty()) {
//     mCentroidsPerSubs.resize(mSubspaceNum);

//     for (int s = 0; s < mSubspaceNum; ++s) {
//       const int sampleSize = std::min(N, int(1e7)); // consider reducing if memory heavy
//       const int startCol = s * mSubsLen;

//       if (verbose) {
//         std::cout << "[PQ] subspace " << s
//                   << " kmeans (sampleSize=" << sampleSize
//                   << ", dsub=" << mSubsLen << ")\n";
//       }

//       // pick sample indices
//       std::vector<int> perm = makePerm(N);

//       // Build arma data: dsub x sampleSize
//       arma::fmat data(mSubsLen, sampleSize);

//       if(sampleSize < N) {
//         for (int i = 0; i < sampleSize; ++i) {
//           const int rid = perm[i];
//           const float* src = XTrain.row(rid).data() + startCol; // subvector start
//           float* dst = data.colptr(i);
//           std::memcpy(dst, src, sizeof(float) * mSubsLen);
//         }
//       } else {
//         #pragma omp parallel for schedule(static)
//         for (int i = 0; i < sampleSize; ++i) {
//           const float* src = XTrain.row(i).data() + startCol;
//           float* dst = data.colptr(i);
//           std::memcpy(dst, src, sizeof(float) * mSubsLen);
//         }
//       }

//       // arma means: dsub x K
//       arma::fmat means(mSubsLen, mCentroidsNum);

      
//       bool ok = arma::kmeans(means, data, mCentroidsNum, arma::static_subset, 50, false);
//       if (!ok) {
//         std::cout << "pq kmeans arma failed\n";
//         assert(false);
//       }

//       // Copy back to Eigen: want (K x dsub), row k = centroid k
//       mCentroidsPerSubs[s].resize(mCentroidsNum, mSubsLen);
//       for (int k = 0; k < mCentroidsNum; ++k) {
//         const float* src = means.colptr(k); // length dsub
//         float* dst = mCentroidsPerSubs[s].row(k).data();
//         std::memcpy(dst, src, sizeof(float) * mSubsLen);
//       }
//     }
//   }

//   mIVF.coarseNorm2 = mIVF.coarseCentroids.rowwise().squaredNorm();

// }

void PQ::trainIVF(RowMatrixXf &XTrain, const int nlist, bool verbose) {
  this->mIVF.mIVFLists = nlist;

  const int N   = static_cast<int>(XTrain.rows());
  const int dim = static_cast<int>(XTrain.cols());

  omp_set_num_threads(64);

  if (mCentroidsNum > N) {
    std::cout << "#Centroids <= #Sample, impossible for KMeans. "
              << mCentroidsNum << " should <= " << N << "\n";
    assert(false);
  }
  if (mIVF.mIVFLists <= 0) {
    std::cout << "Invalid mIVFLists: " << mIVF.mIVFLists << " (N=" << N << ")\n";
    assert(false);
  }

  if (N / mIVF.mIVFLists <= 64) {
    mIVF.mIVFLists = std::max(1, N / 64);
  }

  codebookRow = N;

  // ------------------------------------------------------------
  // 0) dim 必须能被 mSubspaceNum 整除（你说已 padding）
  // ------------------------------------------------------------
  if (dim % mSubspaceNum != 0) {
    std::cout << "[trainIVF] ERROR: XTrain.cols()=" << dim
              << " is not divisible by mSubspaceNum=" << mSubspaceNum << "\n";
    assert(false);
  }
  mSubsLen = dim / mSubspaceNum;

  // ------------------------------------------------------------
  // 1) Shared sample for BOTH IVF coarse and PQ training
  //    - 不随机：直接取前 M 个向量
  // ------------------------------------------------------------
  // const int M = std::min(N, static_cast<int>(1e5)); // 你原先 PQ 的 sample 上限
  const int M = N;
  if (verbose) {
    std::cout << "[trainIVF] Shared sampleSize=" << M
              << " (take first " << M << " vectors)\n";
  }

  // ------------------------------------------------------------
  // 2) 一次性转置 sample：XtS = dim x M (ColMajor), 每列一个样本
  //    之后：
  //    - coarse kmeans: zero-copy 喂给 arma
  //    - PQ: 从 XtS 每列抽取 subspace 段进行 pack
  // ------------------------------------------------------------
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> XtS =
      XTrain.topRows(M).transpose().eval();  // dim x M

  // ------------------------------------------------------------
  // (A) Train IVF coarse centroids using SAME sample (first M)
  // ------------------------------------------------------------
  if (mIVF.nlist == 0 || mIVF.coarseCentroids.rows() == 0) {
    mIVF.nlist = mIVF.mIVFLists;
    mIVF.coarseCentroids.resize(mIVF.mIVFLists, dim);

    if (verbose) {
      std::cout << "[IVF] coarse kmeans: nlist=" << mIVF.mIVFLists
                << " dim=" << dim << " M=" << M << "\n";
    }

    // zero-copy: XtS 是 col-major 且列=样本，完全匹配 arma::kmeans 输入期望
    arma::fmat data(XtS.data(), dim, M, /*copy_aux_mem=*/false, /*strict=*/true);

    arma::fmat means(dim, mIVF.mIVFLists);
    bool ok = arma::kmeans(means, data, mIVF.mIVFLists,
                           arma::static_subset, 50, false);
    if (!ok) {
      std::cout << "coarse kmeans arma failed\n";
      assert(false);
    }
    if (verbose) std::cout << "[IVF] Finish coarse kmeans\n";

    // write back: coarseCentroids (nlist x dim), row c = centroid c
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < mIVF.mIVFLists; ++c) {
      const float* src = means.colptr(c); // length dim
      float* dst = mIVF.coarseCentroids.row(c).data();
      std::memcpy(dst, src, sizeof(float) * dim);
    }
    if (verbose) std::cout << "[IVF] Finish coarse writeback\n";
  }

  // init lists (filled later in add())
  mIVF.lists.clear();
  mIVF.lists.resize(mIVF.nlist);

  // ------------------------------------------------------------
  // (B) Train global PQ codebook (shared by all lists)
  //     For each subspace s:
  //       - pack data_s : (mSubsLen x M)  (列=样本)
  //       - kmeans => means_s : (mSubsLen x K)
  //
  // NOTE: 这里仍然必须 pack，
  // 因为 arma::kmeans 需要紧密矩阵，不能用 XtS 的 row-block view（stride=dim）。
  // ------------------------------------------------------------
  if (mCentroidsPerSubs.empty()) {
    mCentroidsPerSubs.resize(mSubspaceNum);

    if (verbose) {
      std::cout << "[PQ] Train global PQ codebook: M=" << M
                << " subspaces=" << mSubspaceNum
                << " dsub=" << mSubsLen
                << " K=" << mCentroidsNum << "\n";
    }

    // 复用缓冲区，避免每个 subspace 频繁分配
    arma::fmat subData(mSubsLen, M);
    arma::fmat subMeans(mSubsLen, mCentroidsNum);

    for (int s = 0; s < mSubspaceNum; ++s) {
      const int startCol = s * mSubsLen;

      if (verbose) {
        std::cout << "[PQ] subspace " << s
                  << " kmeans (M=" << M
                  << ", dsub=" << mSubsLen
                  << ", startCol=" << startCol << ")\n";
      }

      // pack: subData(:, j) = XtS(startCol : startCol+mSubsLen-1, j)
      #pragma omp parallel for schedule(static)
      for (int j = 0; j < M; ++j) {
        const float* src = XtS.col(j).data() + startCol; // 列内连续
        float* dst = subData.colptr(j);
        std::memcpy(dst, src, sizeof(float) * mSubsLen);
      }

      bool ok = arma::kmeans(subMeans, subData, mCentroidsNum,
                             arma::static_subset, 50, false);
      if (!ok) {
        std::cout << "pq kmeans arma failed (subspace " << s << ")\n";
        assert(false);
      }

      // write back: mCentroidsPerSubs[s] (K x dsub), row k = centroid k
      mCentroidsPerSubs[s].resize(mCentroidsNum, mSubsLen);

      #pragma omp parallel for schedule(static)
      for (int k = 0; k < mCentroidsNum; ++k) {
        const float* src = subMeans.colptr(k); // length dsub
        float* dst = mCentroidsPerSubs[s].row(k).data();
        std::memcpy(dst, src, sizeof(float) * mSubsLen);
      }
    }
  }

  // precompute coarse norm^2
  mIVF.coarseNorm2 = mIVF.coarseCentroids.rowwise().squaredNorm();
}


static inline float l2sqr_avx2(const float* a, const float* b, int d) {
  __m256 acc = _mm256_setzero_ps();
  int i = 0;

  for (; i + 7 < d; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);
    acc = _mm256_fmadd_ps(diff, diff, acc);
  }

  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);

  float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3]
            + tmp[4] + tmp[5] + tmp[6] + tmp[7];

  for (; i < d; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

static inline uint16_t encodeSubspaceNN(const float* xs,
                                       const RowMatrixXf& centroids, // (K x dsub)
                                       int K, int dsub) {
  int best = 0;
  float bestDist = std::numeric_limits<float>::infinity();
  for (int k = 0; k < K; ++k) {
    const float* ck = centroids.row(k).data();
    float dist = l2sqr_avx2(xs, ck, dsub);
    if (dist < bestDist) { bestDist = dist; best = k; }
  }
  return (uint16_t)best;
}

static inline void setPackedBit(std::vector<uint64_t>& bits, size_t bitIndex, uint64_t v01) {
  const size_t word = bitIndex >> 6;      // /64
  const size_t off  = bitIndex & 63;      // %64
  const uint64_t mask = (uint64_t)1 << off;
  bits[word] = (bits[word] & ~mask) | (-(int64_t)v01 & mask);
}


// ==========================================
// encodeIVF: Faiss IVF-PQ style (global PQ)
// ==========================================
void PQ::encodeIVF(const RowMatrixXf &XTrain, bool verbose /*=false*/) {
  const int Nb  = static_cast<int>(XTrain.rows());
  const int dim = static_cast<int>(XTrain.cols());

  // ---- Preconditions ----
  assert(mIVF.nlist > 0);
  assert(mIVF.coarseCentroids.rows() == mIVF.nlist);
  assert(mIVF.coarseCentroids.cols() == dim);

  assert((int)mCentroidsPerSubs.size() == mSubspaceNum);
  assert(mSubsLen > 0);
  assert(mCentroidsNum > 0 && mCentroidsNum <= 1024);
  assert(dim == mSubspaceNum * mSubsLen && "XTrain must be padded to M*mSubsLen");

  assert((int)mCentroidsPerSubs.size() == mSubspaceNum);
  for (int s = 0; s < mSubspaceNum; ++s) {
    assert(mCentroidsPerSubs[s].rows() == mCentroidsNum);
    assert(mCentroidsPerSubs[s].cols() == mSubsLen);
  }

  // 确保 lists 已经存在
  if ((int)mIVF.lists.size() != mIVF.nlist) {
    mIVF.lists.clear();
    mIVF.lists.resize(mIVF.nlist);
  }

  // 可选：清空旧的 index 内容（如果你想 incremental add，删掉这段并改成 append）
  for (auto &lst : mIVF.lists) {
    lst.idmap.clear();
    lst.codes.resize(0, mSubspaceNum);
  }

  // 1) assign each vector to a coarse list
  std::vector<int> assign(Nb);
  const float* coarse = mIVF.coarseCentroids.data();

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < Nb; ++i) {
    assign[i] = assignCoarse1(XTrain.row(i).data(), coarse, mIVF.nlist, dim);
  }

  // 2) count sizes
  std::vector<int> counts(mIVF.nlist, 0);
  for (int i = 0; i < Nb; ++i) counts[assign[i]]++;

  // 3) allocate per-list storage
  for (int lid = 0; lid < mIVF.nlist; ++lid) {
    const int R = counts[lid];
    auto &lst = mIVF.lists[lid];
    lst.idmap.resize(R);
    lst.codes.resize(R, mSubspaceNum); // (rows=R, cols=M), col-major
  }

  // 4) fill (encode PQ codes using GLOBAL codebook)
  std::vector<int> writePos(mIVF.nlist, 0);
  const int nlist = mIVF.nlist;
  const int T = omp_get_max_threads();

  if (mCentroidsNum <= 256) {
    // for (int i = 0; i < Nb; ++i) {
    //   const int lid = assign[i];
    //   const int pos = writePos[lid]++;

    //   auto &lst = mIVF.lists[lid];

    //   // local -> original id
    //   lst.idmap[pos] = i; // 如果你有外部原始 id，这里换成那个 id

    //   // encode M subspaces
    //   for (int s = 0; s < mSubspaceNum; ++s) {
    //     const float* xs = XTrain.row(i).data() + s * mSubsLen;
    //     uint16_t code = encodeSubspaceNN(xs, mCentroidsPerSubs[s], mCentroidsNum, mSubsLen);
    //     lst.codes(pos, s) = code;
    //   }
    // }

  // 4) fill (encode PQ codes using GLOBAL codebook) -- parallel, no races


  // Pass 4.1) per-thread histogram counts: countsT[tid][lid]
  std::vector<int> countsT(T * nlist, 0);

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    int* localCnt = countsT.data() + tid * nlist;

    #pragma omp for schedule(static)
    for (int i = 0; i < Nb; ++i) {
      const int lid = assign[i];
      localCnt[lid] += 1;
    }
  }

  // Pass 4.2) compute per-thread base offset for each list: baseT[tid][lid]
  std::vector<int> baseT(T * nlist, 0);
  for (int lid = 0; lid < nlist; ++lid) {
    int run = 0;
    for (int t = 0; t < T; ++t) {
      baseT[t * nlist + lid] = run;
      run += countsT[t * nlist + lid];
    }
    // 可选 sanity check：run 应该等于 counts[lid]
    // assert(run == counts[lid]);
  }

  // Pass 4.3) parallel write: each thread writes into its own [base, base+count) range
  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();

    // thread-local cursors per list
    std::vector<int> cursor(nlist);
    for (int lid = 0; lid < nlist; ++lid) {
      cursor[lid] = baseT[tid * nlist + lid];
    }

    #pragma omp for schedule(static)
    for (int i = 0; i < Nb; ++i) {
      const int lid = assign[i];
      const int pos = cursor[lid]++;   // thread-local, no race

      auto &lst = mIVF.lists[lid];

      lst.idmap[pos] = i;

      const float* xrow = XTrain.row(i).data();
      for (int s = 0; s < mSubspaceNum; ++s) {
        const float* xs = xrow + s * mSubsLen;
        uint16_t code = encodeSubspaceNN(xs, mCentroidsPerSubs[s], mCentroidsNum, mSubsLen);
        lst.codes(pos, s) = code;
      }
    }
  }
} else if (mCentroidsNum == 512) {
    for (int i = 0; i < Nb; ++i) {
      const int lid = assign[i];
      const int pos = writePos[lid]++;

      auto &lst = mIVF.lists[lid];

      lst.idmap[pos] = i;

      const int R = lst.codes.rows();

      // init once per list
      if (pos == 0) {
        const int nBlocks = (R + 63) / 64;
        lst.codeHiPacked.assign((size_t)mSubspaceNum * (size_t)nBlocks, 0ULL); // bit8 only

        lst.tailStart = (R & ~63);         // floor(R/64)*64
        lst.tailLen   = R - lst.tailStart; // 0..63

        // allocate tailCodes (tailLen x M), even tailLen==0 is OK
        lst.tailCodes.resize(lst.tailLen, mSubspaceNum);
        // optional: lst.tailCodes.setZero();
      }

      const int tailStart = lst.tailStart;

      // ============ CASE A: tail rows -> store full 9-bit code into tailCodes ============
      if (pos >= tailStart) {
        const int tpos = pos - tailStart; // 0..tailLen-1
        for (int s = 0; s < mSubspaceNum; ++s) {
          const float* xs = XTrain.row(i).data() + s * mSubsLen;
          uint16_t code9 = encodeSubspaceNN(xs, mCentroidsPerSubs[s], mCentroidsNum, mSubsLen); // 0..511
          lst.tailCodes(tpos, s) = code9;
        }
        continue;
      }

      // ============ CASE B: main rows (multiple of 64) -> store low8 + packed bit8 ============
      const int nBlocks = (R + 63) / 64;
      const int blk = pos >> 6;
      const int off = pos & 63;
      const uint64_t bitMask = 1ULL << off;

      for (int s = 0; s < mSubspaceNum; ++s) {
        const float* xs = XTrain.row(i).data() + s * mSubsLen;

        uint16_t code = encodeSubspaceNN(xs, mCentroidsPerSubs[s], mCentroidsNum, mSubsLen); // 0..511
        const uint8_t  lo = (uint8_t)(code & 0xFF);
        const uint64_t b8 = (uint64_t)((code >> 8) & 1ULL);

        lst.codes(pos, s) = lo;

        uint64_t &word = lst.codeHiPacked[(size_t)s * (size_t)nBlocks + (size_t)blk];
        if (b8) word |= bitMask;
        else    word &= ~bitMask;
      }
    }
  } else if (mCentroidsNum == 1024) {
    for (int i = 0; i < Nb; ++i) {
      const int lid = assign[i];
      const int pos = writePos[lid]++;

      auto &lst = mIVF.lists[lid];

      lst.idmap[pos] = i;

      const int R = lst.codes.rows();

      // compute tail region once
      if (pos == 0) {
        const int nBlocks = (R + 63) / 64;
        lst.codeHiPacked.assign((size_t)mSubspaceNum * (size_t)nBlocks, 0ULL);   // bit8
        lst.codeHiPacked9.assign((size_t)mSubspaceNum * (size_t)nBlocks, 0ULL);  // bit9

        lst.tailStart = (R & ~63);          // floor(R/64)*64
        lst.tailLen   = R - lst.tailStart;  // 0..63

        if (lst.tailLen > 0) {
          // tailCodes: (tailLen x M), col-major
          lst.tailCodes = ColMatrix<uint16_t>(lst.tailLen, mSubspaceNum);
        } else {
          // optional: clear
          lst.tailCodes = ColMatrix<uint16_t>(0, mSubspaceNum);
        }
      }

      const int tailStart = lst.tailStart;

      // full-block area parameters
      const int nBlocks = (R + 63) / 64;
      const int blk = pos >> 6;
      const int off = pos & 63;
      const uint64_t bitMask = 1ULL << off;

      // ============ CASE A: tail rows -> write full 10-bit into tailCodes ============
      if (pos >= tailStart) {
        const int tpos = pos - tailStart; // 0..tailLen-1
        // store all subspaces codes (10-bit) into tailCodes
        for (int s = 0; s < mSubspaceNum; ++s) {
          const float* xs = XTrain.row(i).data() + s * mSubsLen;
          uint16_t code10 = encodeSubspaceNN(xs, mCentroidsPerSubs[s], mCentroidsNum, mSubsLen); // 0..1023
          lst.tailCodes(tpos, s) = code10;
        }
        continue;
      }

      // ============ CASE B: main rows (multiple of 64) -> write low8 + packed bits ============
      for (int s = 0; s < mSubspaceNum; ++s) {
        const float* xs = XTrain.row(i).data() + s * mSubsLen;

        uint16_t code = encodeSubspaceNN(xs, mCentroidsPerSubs[s], mCentroidsNum, mSubsLen); // 0..1023

        const uint8_t  lo = (uint8_t)(code & 0xFF);
        const uint64_t b8 = (uint64_t)((code >> 8) & 1U);
        const uint64_t b9 = (uint64_t)((code >> 9) & 1U);

        lst.codes(pos, s) = lo;

        uint64_t &w8 = lst.codeHiPacked[(size_t)s * (size_t)nBlocks + (size_t)blk];
        uint64_t &w9 = lst.codeHiPacked9[(size_t)s * (size_t)nBlocks + (size_t)blk];

        if (b8) w8 |= bitMask; else w8 &= ~bitMask;
        if (b9) w9 |= bitMask; else w9 &= ~bitMask;
      }
    }
  }


  // 5) sanity checks
  // for (int lid = 0; lid < mIVF.nlist; ++lid) {
  //   assert(writePos[lid] == counts[lid]);
  // }

  if (verbose) {
    long long tot = 0;
    int nonEmpty = 0;
    for (int lid = 0; lid < mIVF.nlist; ++lid) {
      int r = (int)mIVF.lists[lid].idmap.size();
      tot += r;
      nonEmpty += (r > 0);
    }
    std::cout << "[encodeIVF] Nb=" << Nb
              << " nlist=" << mIVF.nlist
              << " nonEmpty=" << nonEmpty
              << " totalStored=" << tot
              << std::endl;
  }
}

void PQ::encode(const RowMatrixXf &XTrain) {
    mXTrainRows = XTrain.rows();
    mXTrainCols = XTrain.cols();
    mCodebook.resize(mXTrainRows, mSubspaceNum);
    encodeImpl(XTrain, mCodebook);
    prepareSmallCodebook();
}

void PQ::encode(const RowMatrixXf &XTrain, std::vector<int> &toOriginalID, const float alpha) {
    mXTrainRows = XTrain.rows();
    mXTrainCols = XTrain.cols();
    mCodebook.resize(mXTrainRows, mSubspaceNum);
    encodeImpl(XTrain, mCodebook);
    //encodeImplReorder(XTrain, toOriginalID, mCodebook, 64, alpha, 64);
    prepareSmallCodebook();
}


void PQ::buildGroups() {
  constexpr int th = 128;
  const auto &globalSmallCodebook = this->mSmallCodebook;
  const int N = static_cast<int>(globalSmallCodebook.rows());
  const int M = static_cast<int>(globalSmallCodebook.cols());
  assert(M >= 8 && "Need at least 8 subspaces (cols >= 8).");

  // 0) 清空旧内容（257 组都清）
  for (auto& g : groups) {
    g.mSmallCodebook.resize(0, M);
    g.pqID.clear();
  }

  for (int gid = 0; gid < 256; ++gid) {
    for (int s = 0; s < 8; ++s) {
      // bit s == 0 -> low64
      groups[gid].isLow[s] = (((gid >> s) & 1) == 0);
    }
  }

  auto groupIdOfRow = [&](int r) -> uint8_t {
    uint8_t gid = 0;
    for (int s = 0; s < 8; ++s) {
      const uint8_t code = static_cast<uint8_t>(globalSmallCodebook(r, s));
      gid |= static_cast<uint8_t>((code >= 64) ? (1u << s) : 0u);
    }
    return gid; // 0..255
  };

  // 1) 第一遍：统计每组原始大小
  std::array<int, 256> counts{};
  counts.fill(0);
  for (int r = 0; r < N; ++r) {
    const uint8_t gid = groupIdOfRow(r);
    counts[gid]++;
  }

  // 2) 计算每组“保留数量”和“搬走数量”，并统计最后一组大小
  std::array<int, 256> keepCounts{};
  std::array<int, 256> moveCounts{};
  keepCounts.fill(0);
  moveCounts.fill(0);


  int lastCount = 0; // groups[256] 的最终大小

  for (int gid = 0; gid < 256; ++gid) {
    const int sz = counts[gid];

    if (sz < th) {
      // 规则 1：整组搬到最后一组
      keepCounts[gid] = 0;
      moveCounts[gid] = sz;
    } else {
      // 规则 2：保留能整除32的部分，尾巴搬走
      const int keep = (sz / 32) * 32;
      keepCounts[gid] = keep;
      moveCounts[gid] = sz - keep;
    }

    lastCount += moveCounts[gid];
  }

  // 3) 分配各组矩阵 & pqID（0..255 用 keepCounts，256 用 lastCount）
  for (int gid = 0; gid < 256; ++gid) {
    const int keep = keepCounts[gid];
    if (keep == 0) continue;
    groups[gid].mSmallCodebook.resize(keep, M);
    groups[gid].pqID.resize(keep);
  }

  if (lastCount > 0) {
    groups[256].mSmallCodebook.resize(lastCount, M);
    groups[256].pqID.resize(lastCount);
  }

  // 4) 第二遍：按“组内出现顺序”决定去留/搬运，并填充
  std::array<int, 256> seenInGroup{};   // 该行在所属组内是第几个（0-based）
  std::array<int, 256> writeKeepPos{};  // 写入各组(0..255)的位置
  seenInGroup.fill(0);
  writeKeepPos.fill(0);
  int writeLastPos = 0;

  for (int r = 0; r < N; ++r) {
    const uint8_t gid = groupIdOfRow(r);

    const int idxInGroup = seenInGroup[gid]++; // 0..counts[gid)-1
    const bool keepHere = (idxInGroup < keepCounts[gid]);

    if (keepHere) {
      const int pos = writeKeepPos[gid]++;

      for (int c = 0; c < M; ++c) {
        groups[gid].mSmallCodebook(pos, c) = globalSmallCodebook(r, c);
      }
      groups[gid].pqID[pos] = r;
    } else {
      // 搬到最后一组 groups[256]
      const int pos = writeLastPos++;

      for (int c = 0; c < M; ++c) {
        groups[256].mSmallCodebook(pos, c) = globalSmallCodebook(r, c);
      }
      groups[256].pqID[pos] = r;
    }
  }

  // 5) sanity check
  for (int gid = 0; gid < 257; ++gid) {
    for (int k = 0; k < groups[gid].pqID.size(); ++k) {
      int r = groups[gid].pqID[k];
      for (int c = 0; c < M; ++c) {
        assert(groups[gid].mSmallCodebook(k, c) == mSmallCodebook(r, c));
      }
    }
  }
  for (int gid = 0; gid < 256; ++gid) {
    assert(writeKeepPos[gid] == keepCounts[gid]);
    assert(seenInGroup[gid] == counts[gid]);
  }
  assert(writeLastPos == lastCount);

  for(int gid = 0; gid <= 256;++gid){
    printf("%d ", int(groups[gid].mSmallCodebook.rows()));
  }
  puts("");
}


template<class T>
void PQ::encodeImpl(const RowMatrixXf &XTrain, T &codebook) {

  // For each subspace
  for (int i=0; i<mSubspaceNum; i++) {
    // for each row
    #pragma omp parallel for
    for (int rowIdx=0; rowIdx<mXTrainRows; rowIdx++) {
      CodewordType bestCode = 0;
      float bsf = std::numeric_limits<float>::max();
      for (int code=0; code < mCentroidsNum ; code++) {
        float dist = (XTrain.block(rowIdx, i * mSubsLen, 1, mSubsLen) - mCentroidsPerSubs[i].block(code, 0, 1, mSubsLen)).squaredNorm();

        if (dist < bsf) {
          bestCode = static_cast<CodewordType>(code);
          bsf = dist;
        }
      }
      codebook(rowIdx, i) = bestCode;
    }
  }
}

float PQ::subspaceDistSq(
    const RowMatrixXf& XTrain,
    int rowIdx,
    int sIdx,
    int subsLen,
    const Eigen::MatrixXf& centroids,  // [Ks, subsLen]
    int code
) {
    // XTrain.block(rowIdx, sIdx*subsLen, 1, subsLen) - centroids.row(code)
    // squaredNorm
    return (XTrain.block(rowIdx, sIdx * subsLen, 1, subsLen) - centroids.block(code, 0, 1, subsLen)).squaredNorm();
}

// codebook(row, s) 可读写的泛型
template<class T>
void PQ::encodeImplReorder(
    const RowMatrixXf& XTrain,
    std::vector<int> &toOriginalID,
    T& codebook,
    int topC,
    float alpha,
    int codeThreshold
) {
    const int N  = (int)XTrain.rows();
    const int M  = (int)mSubspaceNum;
    const int Ks = (int)mCentroidsNum;
    const int L  = (int)mSubsLen;
    perm_sub.resize(M);
    for(int i =0; i<M;++i) {
      perm_sub[i] = i;
    }

    // -----------------------------
    // Step 1) first encode (原始 encode)
    // -----------------------------
    for (int s = 0; s < M; ++s) {
        #pragma omp parallel for
        for (int row = 0; row < N; ++row) {
            CodewordType bestCode = 0;
            float bsf = std::numeric_limits<float>::max();

            for (int code = 0; code < Ks; ++code) {
                float dist = subspaceDistSq(XTrain, row, s, L, mCentroidsPerSubs[s], code);
                if (dist < bsf) {
                    bsf = dist;
                    bestCode = (CodewordType)code;
                }
            }
            codebook(row, s) = bestCode;
        }
    }

    // -----------------------------
    // Step 2) per-subspace frequency counts
    // counts[s][code]
    // -----------------------------
    std::vector<std::vector<int>> counts(M, std::vector<int>(Ks, 0));

    for (int s = 0; s < M; ++s) {
        for (int row = 0; row < N; ++row) {
            int c = (int)codebook(row, s);
            counts[s][c] += 1;
        }
    }

    // -----------------------------
    // Step 3) relabel codes: freq high -> small code
    // Build perm_code and inv_perm_code
    // perm_code[s][new_code] = old_code
    // inv_perm_code[s][old_code] = new_code
    // Then remap codebook in-place via inv_perm_code
    // -----------------------------
    std::vector<std::vector<int>> perm_code(M, std::vector<int>(Ks, 0));
    std::vector<std::vector<int>> inv_perm_code(M, std::vector<int>(Ks, 0));

    for (int s = 0; s < M; ++s) {
        std::vector<int> codes(Ks);
        std::iota(codes.begin(), codes.end(), 0);

        std::sort(codes.begin(), codes.end(), [&](int a, int b) {
            if (counts[s][a] != counts[s][b]) return counts[s][a] > counts[s][b]; // freq desc
            return a < b; // tie-break
        });

        // new_code -> old_code
        for (int newc = 0; newc < Ks; ++newc) {
            perm_code[s][newc] = codes[newc];
        }
        // old_code -> new_code
        for (int newc = 0; newc < Ks; ++newc) {
            inv_perm_code[s][perm_code[s][newc]] = newc;
        }

        // remap codebook: old -> new (more frequent => smaller)
        for (int row = 0; row < N; ++row) {
            int oldc = (int)codebook(row, s);
            int newc = inv_perm_code[s][oldc];
            codebook(row, s) = (CodewordType)newc;
        }
    }

    // 注意：如果你后续用 code 去索引 centroid（mCentroidsPerSubs[s].row(code)）
    // 那你要把 centroid 的行也按同样 permutation 重新排列，
    // 否则 “code 语义”就变了。
    // 这里我们直接把 centroid 行也重排，使 code->centroid 一致。
    for (int s = 0; s < M; ++s) {
        Eigen::MatrixXf newC(Ks, L);
        for (int newc = 0; newc < Ks; ++newc) {
            int oldc = perm_code[s][newc];
            newC.row(newc) = mCentroidsPerSubs[s].row(oldc);
        }
        mCentroidsPerSubs[s] = std::move(newC);
        // counts 也要同步变成 new-code 顺序，方便后面 topC
        std::vector<int> newCounts(Ks, 0);
        for (int newc = 0; newc < Ks; ++newc) {
            int oldc = perm_code[s][newc];
            newCounts[newc] = counts[s][oldc];
        }
        counts[s] = std::move(newCounts);
    }

    // -----------------------------
    // Step 4) adjustment:
    // for each vector/subspace:
    // if code > codeThreshold:
    //   among topC codes (0..topC-1), pick one with minimal error
    //   if bestErr < alpha * origErr, replace code
    // -----------------------------
    const int C = std::min(topC, Ks);
    if(alpha > 1.0) {
      for (int s = 0; s < M; ++s) {
          #pragma omp parallel for
          for (int row = 0; row < N; ++row) {
              int curCode = (int)codebook(row, s);
              if (curCode < codeThreshold) continue;

              float origErr = subspaceDistSq(XTrain, row, s, L, mCentroidsPerSubs[s], curCode);

              int bestCode = curCode;
              float bestErr = std::numeric_limits<float>::max();

              // only check topC most frequent (now they are 0..C-1)
              for (int cand = 0; cand < C; ++cand) {
                  float e = subspaceDistSq(XTrain, row, s, L, mCentroidsPerSubs[s], cand);
                  if (e < bestErr) {
                      bestErr = e;
                      bestCode = cand;
                  }
              }

              if (bestCode != curCode && bestErr < alpha * origErr) {
                  codebook(row, s) = (CodewordType)bestCode;
              }
          }
      }
    }

    // -----------------------------
    // Step 5) reorder subspaces:
    // "subspaces按topC频率和最高的降序排序"
    // 这里用 (topCMass, maxFreq) desc 作为排序键
    // -----------------------------
    std::vector<SubspaceScore> scores;
    scores.reserve(M);
    for (int s = 0; s < M; ++s) {
        double topCMass = 0.0;
        for (int c = 0; c < C; ++c) topCMass += (double)counts[s][c];
        topCMass /= (double)N;

        int maxCnt = *std::max_element(counts[s].begin(), counts[s].end());
        double maxFreq = (double)maxCnt / (double)N;

        scores.push_back(SubspaceScore{s, topCMass, maxFreq});
    }

    std::sort(scores.begin(), scores.end(), [](const SubspaceScore& a, const SubspaceScore& b) {
        if (a.topCMass != b.topCMass) return a.topCMass > b.topCMass;
        if (a.maxFreq  != b.maxFreq)  return a.maxFreq  > b.maxFreq;
        return a.s < b.s;
    });

    // perm_sub[new_pos] = old_s
    perm_sub.resize(M);
    for (int newPos = 0; newPos < M; ++newPos) perm_sub[newPos] = scores[newPos].s;

    // Apply subspace reorder to:
    // 1) codebook columns
    // 2) mCentroidsPerSubs
    // 3) counts
    {
        // codebook: create new view/matrix via temp
        // 这里用一个临时二维容器写回（你可改成 Eigen Matrix/自家矩阵）
        std::vector<CodewordType> tmp((size_t)N * (size_t)M);

        for (int row = 0; row < N; ++row) {
            for (int newS = 0; newS < M; ++newS) {
                int oldS = perm_sub[newS];
                tmp[(size_t)row * M + newS] = (CodewordType)codebook(row, oldS);
            }
        }
        for (int row = 0; row < N; ++row)
            for (int s = 0; s < M; ++s)
                codebook(row, s) = tmp[(size_t)row * M + s];

        // centroids & counts
        auto oldCentroids = mCentroidsPerSubs;
        auto oldCounts = counts;
        for (int newS = 0; newS < M; ++newS) {
            int oldS = perm_sub[newS];
            mCentroidsPerSubs[newS] = oldCentroids[oldS];
            counts[newS] = oldCounts[oldS];
        }
    }

    // -----------------------------
    // Step 6) reorder vectors by dimAlpha
    // dimAlpha = number of first dimAlpha subspaces whose code < topC
    // (理解：从 subspace 0 开始计数连续满足 code < topC 的前缀长度)
    // 然后按 dimAlpha 降序排序
    // -----------------------------
    std::vector<int> dimAlphaVec(N, 0);
    for (int row = 0; row < N; ++row) {
        int da = 0;
        for (int s = 0; s < M; ++s) {
            int c = (int)codebook(row, s);
            if (c < C) da++;
            else break; // “前dimAlpha subspaces”理解为前缀
        }
        dimAlphaVec[row] = da;
    }

    std::vector<int> perm_vec(N);
    std::iota(perm_vec.begin(), perm_vec.end(), 0);
    std::sort(perm_vec.begin(), perm_vec.end(), [&](int a, int b) {
        if (dimAlphaVec[a] != dimAlphaVec[b]) return dimAlphaVec[a] > dimAlphaVec[b];
        return a < b;
    });

    // Apply vector reorder to codebook rows (以及你如果需要也可同步重排 XTrain / 原始 vectors)
  {
      std::vector<CodewordType> tmp((size_t)N * (size_t)M);

      // ✅ 同步维护：newRow -> originalID
      // 假设 PQ 类里有成员 std::vector<int> toOriginalID;
      if ((int)toOriginalID.size() != N) {
          toOriginalID.resize(N);
          std::iota(toOriginalID.begin(), toOriginalID.end(), 0);
      }
      std::vector<int> newToOriginalID(N);

      for (int newRow = 0; newRow < N; ++newRow) {
          int oldRow = perm_vec[newRow];

          // 重排 codebook
          for (int s = 0; s < M; ++s) {
              tmp[(size_t)newRow * M + s] = (CodewordType)codebook(oldRow, s);
          }

          // ✅ 重排 ID 映射
          newToOriginalID[newRow] = toOriginalID[oldRow];
      }

      // 写回 codebook
      for (int row = 0; row < N; ++row)
          for (int s = 0; s < M; ++s)
              codebook(row, s) = tmp[(size_t)row * M + s];

      // ✅ 写回映射
      toOriginalID = std::move(newToOriginalID);
  }

    inv_perm_sub_.assign(M, 0);
    for (int newS = 0; newS < M; ++newS) inv_perm_sub_[perm_sub[newS]] = newS;

    const int batchSize = 32;
    const int batchNum  = N / batchSize;
    pruneMarks.reserve((batchNum * M + 63) / 64);

    long long curWord = 0;
    int bitPos = 0;   // 0..63

    for (int batchId = 0; batchId < batchNum; ++batchId) {
        int i = batchId * batchSize;

        for (int s = 0; s < M; ++s) {
            bool all64 = true;

            for (int j = 0; j < batchSize; ++j) {
                if (codebook(i + j, s) >= 64) {
                    all64 = false;
                    break;
                }
            }

            // ---- 写 1 个 bit（流式）----
            if (all64) {
                curWord |= (1LL << bitPos);  // true -> 1
            }
            // false -> 0（什么都不做）

            bitPos++;

            if (bitPos == 64) {
                pruneMarks.push_back(curWord);
                curWord = 0;
                bitPos = 0;
            }
        }
    }
}

RowMatrixXf PQ::decode() {
  int numSamples = mCodebook.rows();       // 数据样本数量

  RowMatrixXf decodedData(numSamples, mSubspaceNum * mSubsLen);
  // 逐个子空间解码
  for (int iSubs = 0; iSubs < mSubspaceNum; ++iSubs) {
      const RowMatrixXf& centroids = mCentroidsPerSubs[iSubs];  // 当前子空间的质心矩阵
      for (int iSample = 0; iSample < numSamples; ++iSample) {
          int centroidIndex = mCodebook(iSample, iSubs);  // 当前样本的质心索引
          decodedData.block(iSample, iSubs * mSubsLen, 1, mSubsLen) = centroids.row(centroidIndex);
      }
  }

  return decodedData;
}


LabelDistVecF PQ::search(const RowMatrixXf &XTest, const int k, bool verbose) {
  // LUTType lut(mCentroidsNum, mSubspaceNum);
  LabelDistVecF ret;
  // ret.labels.resize(k * XTest.rows());
  // ret.distances.resize(k * XTest.rows());
  // static TopKHeap answers;
  // answers.reset(k);

  // for (int q_idx=0; q_idx < (int)XTest.rows(); q_idx++) {
  //   CreateLUT(XTest.row(q_idx), lut);
  //   searchHeap(lut, k, q_idx, answers);
  // }



  // std::copy(answers.ids, answers.ids+k, ret.labels.begin());

  assert(false && "NOT IMPLEMENTED");


  return ret;
}



// XTest is Queries
// We use PQ to search @Refine candidates, then do exact distance search to find top k, where k < @Refine
LabelDistVecF PQ::refine(const RowMatrixXf &XTest, const LabelDistVecF &answersIn, const RowMatrixXf &XTrain, const int k) {
  int refineNum = answersIn.labels.size() / XTest.rows();
  LabelDistVecF ret;
  ret.labels.resize(XTest.rows() * k);
  ret.distances.resize(XTest.rows() * k);
  f::float_maxheap_t answers = {
    size_t(XTest.rows()), size_t(k), ret.labels.data(), ret.distances.data()
  };
  
  // heapmax
  for (int q_idx=0; q_idx < (int)XTest.rows(); q_idx++) {
    int * __restrict heap_ids = answers.ids + q_idx * k;
    float * __restrict heap_dis = answers.val + q_idx * k;

    f::heap_heapify<f::CMax<float, int>> (k, heap_dis, heap_ids);

    for (int i=0; i<refineNum; i++) {
      float dist = (XTest.row(q_idx) - XTrain.row(answersIn.labels[q_idx * refineNum + i])).squaredNorm();
      if (f::CMax<float, int>::cmp(heap_dis[0], dist)) {
        f::heap_pop<f::CMax<float, int>>(k, heap_dis, heap_ids);
        f::heap_push<f::CMax<float, int>>(k, heap_dis, heap_ids, dist, answersIn.labels[q_idx * refineNum + i]);
      }
    }
    f::heap_reorder<f::CMax<float, int>> (k, heap_dis, heap_ids);
  }

  return ret;
}

void PQ::parseMethodString(std::string methodString) {

  if (methodString.rfind("PQ", 0) == 0) {
    int bitPerCentroid;
    if (std::sscanf(methodString.c_str(), "PQ(%d,%d)", &bitPerCentroid, &mSubspaceNum) == 2) {
      if (bitPerCentroid > sizeof(CodewordType)*8) {
        std::cerr << "bitPerCentroid can NOT be larger than" << sizeof(CodewordType)*8 << "now" << std::endl;
      }
      mCentroidsNum = (1 << bitPerCentroid);
      return;
    }
  }

  std::cerr << "method string parse error with input " << methodString << std::endl;
  exit(0);
}



void PQ::loadMetaData(const std::string &filepath) {
  std::ifstream in(filepath, std::ios::binary);

  loadOneData(this->mCentroidsNum, in);
  loadOneData(this->mSubspaceNum, in);
  loadOneData(this->mSubsLen, in);

  std::cout << "Meta Data of this codebook: " << std::endl
    << "#Centroids per Subspace: " << mCentroidsNum << std::endl
    << "#Subspaces: " << mSubspaceNum << std::endl;

  in.close();
}

void PQ::saveMetaData(const std::string &filepath) {
  std::ofstream out(filepath, std::ios::binary);

  saveOneData(this->mCentroidsNum, out);
  saveOneData(this->mSubspaceNum, out);
  saveOneData(this->mSubsLen, out);
  

  out.close();
}

void PQ::prepareSmallCodebook() {
  mSmallCodebook = ColMatrix<uint8_t>(mCodebook.rows(), mCodebook.cols());
  for (int i = 0; i < mCodebook.rows(); ++i) {
    for (int j = 0; j < mCodebook.cols(); ++j) {
        mSmallCodebook(i, j) = static_cast<uint8_t>(mCodebook(i, j));
    }
  }
}

LabelDistVecF PQ::PQScanSIMD_float_LUT(const RowVectorXf &XTest, const std::vector<int>&ids) {
  constexpr int nSubspaces = 16;

  LabelDistVecF res;
  res.labels = ids;
  res.distances.reserve(ids.size());
  if(lut.size() == 0) {
    lut = LUTType(mCentroidsNum, nSubspaces);
  }
  CreateLUT(XTest);

  const int K = mCentroidsNum;

  alignas(32) static const int32_t base_offsets[16] = {
    0 * K, 1 * K, 2 * K, 3 * K,
    4 * K, 5 * K, 6 * K, 7 * K,
    8 * K, 9 * K, 10 * K, 11 * K,
    12 * K, 13 * K, 14 * K, 15 * K
  };

  uint16_t* codes = mCodebook.data();

  for (int id : ids) {
      if(id == -1) {
        res.distances.push_back(std::numeric_limits<float>::max());
        continue;
      }
      
      __m128i code_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes));       // 16 x uint8
      __m256i code_lo_32 = _mm256_cvtepu16_epi32(code_lo);                             // 8 x int32
      __m256i code_hi_32 = _mm256_cvtepu16_epi32(_mm_srli_si128(code_lo, 8));         // upper 8
      
      __m256i base_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_offsets));
      __m256i base_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_offsets + 8));
      
      __m256i offset0 = _mm256_add_epi32(base_lo, code_lo_32);
      __m256i offset1 = _mm256_add_epi32(base_hi, code_hi_32);
      
      __m256 gathered0 = _mm256_i32gather_ps(lut.data(), offset0, 4);
      __m256 gathered1 = _mm256_i32gather_ps(lut.data(), offset1, 4);
      
      // Horizontal sum
      __m128 sum0 = _mm_add_ps(_mm256_castps256_ps128(gathered0), _mm256_extractf128_ps(gathered0, 1));
      __m128 sum1 = _mm_add_ps(_mm256_castps256_ps128(gathered1), _mm256_extractf128_ps(gathered1, 1));
      __m128 sum = _mm_add_ps(sum0, sum1);
      sum = _mm_hadd_ps(sum, sum);
      sum = _mm_hadd_ps(sum, sum);
      res.distances.push_back(_mm_cvtss_f32(sum));

      codes += nSubspaces;
    }

  return res;
}


template<int nSubspaces>
LabelDistVecF PQ::computeFloatDistImpl(const RowVectorXf &XTest, const std::vector<int>&ids) {                                                               

  LabelDistVecF res;
  res.labels = ids;
  res.distances.reserve(ids.size());
  if(lut.size() == 0) {
    lut = LUTType(mCentroidsNum, nSubspaces);
  }
  CreateLUT(XTest);

  const int ksub = lut.rows();
  for(int id : ids) {
    // heap not be pushed at least topk times
    if(id == -1){
      res.distances.push_back(std::numeric_limits<float>::max());
      continue;
    }


    float dist = 0.;
    const float * luts = lut.data();
    // mSmallcodebook is col-major, not suitable for this
    uint16_t* codes = mCodebook.data() + id*mCodebook.cols();
    for (int col=0; col < nSubspaces; col++) {
      dist += luts[*codes];
      codes++;
      luts += ksub;
    }
    // const uint8_t* codes = mSmallCodebook.data() + id * nSubspaces;
    // for (int col = 0; col < nSubspaces; col++) {
    //     dist += luts[codes[col]];
    //     luts += ksub;
    // }
    res.distances.push_back(dist);
  }

  return res;
}

LabelDistVecF PQ::computeFloatDist(const RowVectorXf &XTest, const std::vector<int>&ids) {   

  switch(mSubspaceNum){
    case 2: return computeFloatDistImpl<2>(XTest, ids);
    case 3: return computeFloatDistImpl<3>(XTest, ids);
    case 4: return computeFloatDistImpl<4>(XTest, ids);
    case 6: return computeFloatDistImpl<6>(XTest, ids);
    case 8: return computeFloatDistImpl<8>(XTest, ids);
    case 10: return computeFloatDistImpl<10>(XTest, ids);
    case 12: return computeFloatDistImpl<12>(XTest, ids);
    case 14: return computeFloatDistImpl<14>(XTest, ids);
    case 15: return computeFloatDistImpl<15>(XTest, ids);
    case 16: return computeFloatDistImpl<16>(XTest, ids);
    case 18: return computeFloatDistImpl<18>(XTest, ids);
    case 20: return computeFloatDistImpl<20>(XTest, ids);
    case 22: return computeFloatDistImpl<22>(XTest, ids);
    case 24: return computeFloatDistImpl<24>(XTest, ids);
    case 26: return computeFloatDistImpl<26>(XTest, ids);
    case 28: return computeFloatDistImpl<28>(XTest, ids);
    case 30: return computeFloatDistImpl<30>(XTest, ids);
    case 32: return computeFloatDistImpl<32>(XTest, ids);
    case 36: return computeFloatDistImpl<36>(XTest, ids);
    default:
      switch(mSubspaceNum){
        default:
          case 5: return computeFloatDistImpl<5>(XTest, ids);
          case 7: return computeFloatDistImpl<7>(XTest, ids);
          case 9: return computeFloatDistImpl<9>(XTest, ids);
          case 11: return computeFloatDistImpl<11>(XTest, ids);
          case 13: return computeFloatDistImpl<13>(XTest, ids);
          case 15: return computeFloatDistImpl<15>(XTest, ids);
          case 17: return computeFloatDistImpl<17>(XTest, ids);
          std::cerr << "You could add #Subspaces you want to support here" << std::endl;
          assert(false);
      }
      
  }
}

// struct Distribution{
//   std::vector<std::vector<double>> dists;
//   double min, top100min, top200min, top1000min;
//   double all1percent, all5percent, all10percent, mean, median, max;
//   double qmax, qmin, lutMean, lutMedian;
// };
PQ::Distribution PQ::getFloatDistDistribution(const RowMatrixXf &XTest) {
  Distribution distribution;

  const int numRows = mCodebook.rows();
  const int numTest = XTest.rows();
  distribution.dists.resize(numTest);
  distribution.quantizedDists.resize(numTest);

  for(int qIdx = 0; qIdx < numTest; ++qIdx) {
    RowVectorXf curTest = XTest.row(qIdx);
    distribution.dists[qIdx].resize(numRows);
    distribution.quantizedDists[qIdx].resize(numRows);

    lut = LUTType(mCentroidsNum, mSubspaceNum);
    CreateLUT(curTest);
    DataQuantizer quantizer;
    quantizer.trainQuick(lut, mSubspaceNum);
    auto quantizedLUT = quantizer.quantize<uint8_t>(lut);
    // codebook is row-major
    CodewordType* codes = mCodebook.data();
    const int ksub = lut.rows();
    const int qsub = quantizedLUT.rows();
    #pragma omp parallel for
    for (int i = 0; i < numRows; i++) {
        double dist = 0.0;
        uint8_t quantizedDist = 0;
        const float* luts = lut.data();
        const uint8_t* quantizedLuts = quantizedLUT.data();
        CodewordType* cur = codes + i * mSubspaceNum;

        for (int col = 0; col < mSubspaceNum; col++) {
            dist += luts[cur[col]];
            luts += ksub;
            quantizedDist += quantizedLuts[cur[col]];
            quantizedLuts += qsub;
        }

        distribution.dists[qIdx][i] = dist;
        distribution.quantizedDists[qIdx][i] = quantizedDist;
    }

    std::sort(std::execution::par, distribution.dists[qIdx].begin(), distribution.dists[qIdx].end());
    std::sort(std::execution::par, distribution.quantizedDists[qIdx].begin(), distribution.quantizedDists[qIdx].end());

    assert(distribution.dists[qIdx].size() >= 1000);
    distribution.min += distribution.dists[qIdx][0] / numTest;
    distribution.top100min += distribution.dists[qIdx][100-1] / numTest;
    distribution.top200min += distribution.dists[qIdx][200-1] / numTest;
    distribution.top1000min += distribution.dists[qIdx][1000-1] / numTest;
    
    auto getPPercent = [&distribution,qIdx](const float P) -> double {
      float pos = P * distribution.dists.size();            
      size_t idx = static_cast<size_t>(pos);   
      float frac = pos - idx;                  

      if (idx == 0) return distribution.dists[qIdx].front();       
      if (idx >= distribution.dists[qIdx].size()) return distribution.dists[qIdx].back();

      return distribution.dists[qIdx][idx - 1] * (1.0f - frac) + distribution.dists[qIdx][idx] * frac;
    };

    distribution.all1percent += getPPercent(0.01) / numTest;
    distribution.all5percent += getPPercent(0.05) / numTest;
    distribution.all10percent += getPPercent(0.10) / numTest;
    distribution.max += distribution.dists[qIdx].back() / numTest;
    distribution.mean = std::reduce(distribution.dists[qIdx].begin(), distribution.dists[qIdx].end(), 0.0) / distribution.dists[qIdx].size();
    size_t n = distribution.dists[qIdx].size();
    double median;
    if (n % 2 == 0)
        median += (distribution.dists[qIdx][n / 2 - 1] + distribution.dists[qIdx][n / 2]) / 2.0;
    else
        median += distribution.dists[qIdx][n / 2];
    distribution.median = median / numTest;
    distribution.qmin += quantizer.qmin / numTest;
    distribution.qmax += quantizer.qmax / numTest;

    Eigen::VectorXf lutFlat = Eigen::Map<Eigen::VectorXf>(lut.data(), lut.size());
    std::sort(lutFlat.data(), lutFlat.data() + lutFlat.size());
    double lutMedian;
    n = lutFlat.size();
    if (n % 2 == 0)
        lutMedian = (lutFlat[n/2 - 1] + lutFlat[n/2]) / 2.0;
    else
        lutMedian = lutFlat[n/2];
    distribution.lutMedian += mSubspaceNum*lutMedian / numTest;
    distribution.lutMean += mSubspaceNum*lut.mean() / numTest;
  }

  return distribution;
}


PQ::SingleDistribution PQ::getSingleDistribution(const RowVectorXf &XTest) {
  static int i = 0;
  const int numRows = mCodebook.rows();
  PQ::SingleDistribution distribution;
  distribution.dists.resize(numRows);
  distribution.quantizedDists.resize(numRows);
  lut = LUTType(mCentroidsNum, mSubspaceNum);
  CreateLUT(XTest);
  DataQuantizer quantizer;
  quantizer.trainQuick(lut, mSubspaceNum);
  auto quantizedLUT = quantizer.quantize<uint8_t>(lut);
  // codebook is row-major
  CodewordType* codes = mCodebook.data();
  const int ksub = lut.rows();
  const int qsub = quantizedLUT.rows();
  #pragma omp parallel for
  for (int i = 0; i < numRows; i++) {
      double dist = 0.0;
      uint8_t quantizedDist = 0;
      const float* luts = lut.data();
      const uint8_t* quantizedLuts = quantizedLUT.data();
      CodewordType* cur = codes + i * mSubspaceNum;

      for (int col = 0; col < mSubspaceNum; col++) {
          dist += luts[cur[col]];
          luts += ksub;
          quantizedDist += quantizedLuts[cur[col]];
          quantizedLuts += qsub;
      }

      distribution.dists[i] = dist;
      distribution.quantizedDists[i] = quantizedDist;
  }

  std::sort(std::execution::par, distribution.dists.begin(), distribution.dists.end());
  std::sort(std::execution::par, distribution.quantizedDists.begin(), distribution.quantizedDists.end());

  return distribution;
}

static inline void selectNProbeL2(
    const RowVectorXf& q,
    const RowMatrixXf& coarseCentroids,   // (nlist x dim)
    const Eigen::VectorXf& coarseNorm2,   // (nlist), precomputed ||c_i||^2
    int nprobe,
    std::vector<int>& out_ids
) {
  const int nlist = static_cast<int>(coarseCentroids.rows());
  const int dim   = static_cast<int>(coarseCentroids.cols());
  assert(dim == q.size());
  assert(coarseNorm2.size() == nlist);

  nprobe = std::min(nprobe, nlist);
  if (nprobe <= 0) { out_ids.clear(); return; }

  // dots[i] = c_i^T q
  Eigen::VectorXf dots = coarseCentroids * q.transpose(); // (nlist)

  const float q2 = q.squaredNorm();
  Eigen::VectorXf dist = coarseNorm2.array() + q2 - 2.0f * dots.array();

  struct Pair { float d; int id; };
  std::vector<Pair> buf(nlist);
  for (int i = 0; i < nlist; ++i) buf[i] = { dist[i], i };

  auto cmp = [](const Pair& a, const Pair& b){ return a.d < b.d; };

  if (nprobe < nlist) {
    std::nth_element(buf.begin(), buf.begin() + nprobe, buf.end(), cmp);
  }
  std::sort(buf.begin(), buf.begin() + nprobe, cmp);

  out_ids.clear();
  out_ids.reserve(nprobe);
  for (int i = 0; i < nprobe; ++i) out_ids.push_back(buf[i].id);
}

#define CASE_PQ_1024CENTROIDS(NS_)                                                \
  case (NS_):                                                                     \
    searchHeapSIMD1024Centroids<(NS_)>(                                           \
        quantizedLUT, k, answers,                                                 \
        const_cast<ColMatrix<uint8_t>&>(lst.codes),                               \
        const_cast<ColMatrix<uint16_t>&>(lst.tailCodes),                                                            \
        lst.codeHiPacked.data(),                                                  \
        lst.codeHiPacked9.data(),                                                 \
        const_cast<std::vector<int>&>(lst.idmap)                                  \
    );                                                                            \
    break

#define CASE_PQ_512CENTROIDS(NS_)                                                 \
  case (NS_):                                                                     \
    searchHeapSIMD512Centroids<(NS_)>(                                            \
        quantizedLUT, k, answers,                                                 \
        const_cast<ColMatrix<uint8_t>&>(lst.codes),                               \
        const_cast<ColMatrix<uint16_t>&>(lst.tailCodes),                          \
        lst.codeHiPacked.data(),                                                  \
        const_cast<std::vector<int>&>(lst.idmap)                                  \
    );                                                                            \
    break

#define CASE_PQ_256CENTROIDS(NS_)                                                 \
  case (NS_):                                                                     \
    searchHeapSIMD256Centroids<(NS_)>(                                            \
        quantizedLUT, k, answers,                                                 \
        const_cast<ColMatrix<uint8_t>&>(lst.codes),                               \
        const_cast<std::vector<int>&>(lst.idmap)                                  \
    );                                                                            \
    break


#define CASE_PQ_128CENTROIDS(NS_)                                                 \
  case (NS_):                                                                     \
    searchHeapSIMD128Centroids<uint8_t, (NS_)>(                                   \
        quantizedLUT, k, 0, answers,                                              \
        const_cast<ColMatrix<uint8_t>&>(lst.codes),                               \
        const_cast<std::vector<int>&>(lst.idmap)                                  \
    );                                                                            \
    break

#define CASE_PQ_64CENTROIDS(NS_)                                                  \
  case (NS_):                                                                     \
    searchHeapSIMDLargeCentroids<uint8_t, 64, (NS_)>(                             \
        quantizedLUT, k, 0, answers,                                              \
        const_cast<ColMatrix<uint8_t>&>(lst.codes),                               \
        const_cast<std::vector<int>&>(lst.idmap)                                  \
    );                                                                            \
    break


#define CASE_PQ_32CENTROIDS(NS_)                                                  \
  case (NS_):                                                                     \
    searchHeapSIMD<uint8_t, 32, (NS_)>(                                           \
        quantizedLUT, k, 0, answers,                                              \
        const_cast<ColMatrix<uint8_t>&>(lst.codes),                               \
        const_cast<std::vector<int>&>(lst.idmap)                                  \
    );                                                                            \
    break


void PQ::searchOneIVF(
    const RowVectorXf &XTest,
    const int k,
    int nprobe,
    bool simd,
    bool verbose,
    TopKHeap &answers,
    int idBias,
    DataQuantizer &quantizer
) {
  this->idBias = idBias;

  ColMatrix<uint8_t> quantizedLUT = quantizer.quantize<uint8_t>(lut);
  // for(int i = 0; i < lut.rows(); ++i) {
  //   for(int j = 0; j < lut.cols(); ++j) {
  //     printf("%f ", float(lut(i, j)));
  //   }
  //   puts("");
  // }

  // for(int i = 0; i < quantizedLUT.rows(); ++i) {
  //   for(int j = 0; j < quantizedLUT.cols(); ++j) {
  //     printf("%d ", int(quantizedLUT(i, j)));
  //   }
  //   puts("");
  // }
  // exit(0);

  // 2) pick nprobe lists by coarse L2
  std::vector<int> probe_ids;
  
  // auto start = std::chrono::steady_clock::now();
  if(mIVF.mIVFLists == 1) {
    probe_ids.push_back(0);
  } else {
    selectNProbeL2(XTest, mIVF.coarseCentroids, mIVF.coarseNorm2, nprobe, probe_ids);
  }
  // auto end = std::chrono::steady_clock::now();
  // lutSec += std::chrono::duration<double>(end - start).count();

  for (int t = 0; t < (int)probe_ids.size(); ++t) {
    const int lid = probe_ids[t];
    const auto& lst = mIVF.lists[lid];

    if (lst.idmap.empty()) continue;
    if (lst.codes.rows() == 0) continue;

    // sanity
    assert((int)lst.idmap.size() == lst.codes.rows());
    assert(lst.codes.cols() == mSubspaceNum);


    // 你当前 kernel 是 uint8 + 128 centroids (按你给的函数名)
    // 这里只给 mSubspaceNum==8 的例子；你可以照你原来的 switch 扩展
#if defined(__AVX512VBMI__) && defined(__AVX512BW__) && defined(__AVX512F__)
    if(mCentroidsNum == 256) {
      switch(mSubspaceNum){
        CASE_PQ_256CENTROIDS(8);
        CASE_PQ_256CENTROIDS(16);
        CASE_PQ_256CENTROIDS(24);
        CASE_PQ_256CENTROIDS(32);
        default:
          assert(false && "Unsupported mSubspaceNum for SIMD256Centroids IVF path");
      }
    } else if (mCentroidsNum == 512) {
      switch(mSubspaceNum){
        CASE_PQ_512CENTROIDS(5);
        CASE_PQ_512CENTROIDS(6);
        CASE_PQ_512CENTROIDS(7);
        CASE_PQ_512CENTROIDS(8);
        CASE_PQ_512CENTROIDS(10);
        CASE_PQ_512CENTROIDS(12);
        CASE_PQ_512CENTROIDS(13);
        CASE_PQ_512CENTROIDS(14);
        CASE_PQ_512CENTROIDS(16);
        CASE_PQ_512CENTROIDS(24);
        default:
          assert(false && "Unsupported mSubspaceNum for SIMD512Centroids IVF path");
      }
    } else if (mCentroidsNum == 1024) {
      switch(mSubspaceNum){
        CASE_PQ_1024CENTROIDS(4);
        CASE_PQ_1024CENTROIDS(5);
        CASE_PQ_1024CENTROIDS(6);
        CASE_PQ_1024CENTROIDS(7);
        CASE_PQ_1024CENTROIDS(8);
        CASE_PQ_1024CENTROIDS(10);
        CASE_PQ_1024CENTROIDS(11);
        CASE_PQ_1024CENTROIDS(12);
        CASE_PQ_1024CENTROIDS(13);
        CASE_PQ_1024CENTROIDS(14);
        CASE_PQ_1024CENTROIDS(16);
        CASE_PQ_1024CENTROIDS(24);
        default:
          assert(false && "Unsupported mSubspaceNum for SIMD1024Centroids IVF path");
      }
    }
#endif

    if(mCentroidsNum == 128){
      switch (mSubspaceNum) {
        CASE_PQ_128CENTROIDS(8);
        CASE_PQ_128CENTROIDS(9);
        CASE_PQ_128CENTROIDS(10);
        CASE_PQ_128CENTROIDS(11);
        CASE_PQ_128CENTROIDS(12);
        CASE_PQ_128CENTROIDS(13);
        CASE_PQ_128CENTROIDS(16);
        CASE_PQ_128CENTROIDS(17);
        CASE_PQ_128CENTROIDS(19);
        CASE_PQ_128CENTROIDS(20);
        CASE_PQ_128CENTROIDS(22);
        CASE_PQ_128CENTROIDS(24);
        CASE_PQ_128CENTROIDS(25);
        CASE_PQ_128CENTROIDS(26);
        CASE_PQ_128CENTROIDS(28);
        CASE_PQ_128CENTROIDS(30);
        CASE_PQ_128CENTROIDS(32);
        CASE_PQ_128CENTROIDS(34);
        CASE_PQ_128CENTROIDS(39);
        CASE_PQ_128CENTROIDS(40);
        CASE_PQ_128CENTROIDS(43);
        CASE_PQ_128CENTROIDS(50);
        CASE_PQ_128CENTROIDS(52);
        CASE_PQ_128CENTROIDS(60);
        CASE_PQ_128CENTROIDS(70);
        CASE_PQ_128CENTROIDS(84);
        CASE_PQ_128CENTROIDS(86);
        CASE_PQ_128CENTROIDS(100);
        CASE_PQ_128CENTROIDS(103);
        CASE_PQ_128CENTROIDS(131);
        CASE_PQ_128CENTROIDS(157);
        CASE_PQ_128CENTROIDS(160);
        CASE_PQ_128CENTROIDS(192);
        CASE_PQ_128CENTROIDS(229);
        CASE_PQ_128CENTROIDS(274);
        CASE_PQ_128CENTROIDS(683);
        CASE_PQ_128CENTROIDS(820);

        default:
          assert(false && "Unsupported mSubspaceNum for SIMD128Centroids IVF path");
      } 
    } else if (mCentroidsNum == 64) {
      switch (mSubspaceNum) {
        CASE_PQ_64CENTROIDS(8);
        CASE_PQ_64CENTROIDS(12);
        CASE_PQ_64CENTROIDS(13);
        CASE_PQ_64CENTROIDS(16);
        CASE_PQ_64CENTROIDS(17);
        CASE_PQ_64CENTROIDS(20);
        CASE_PQ_64CENTROIDS(21);
        CASE_PQ_64CENTROIDS(22);
        CASE_PQ_64CENTROIDS(24);
        CASE_PQ_64CENTROIDS(25);
        CASE_PQ_64CENTROIDS(28);
        CASE_PQ_64CENTROIDS(32);
        CASE_PQ_64CENTROIDS(33);
        CASE_PQ_64CENTROIDS(37);
        CASE_PQ_64CENTROIDS(38);
        CASE_PQ_64CENTROIDS(42);
        CASE_PQ_64CENTROIDS(48);
        CASE_PQ_64CENTROIDS(50);
        CASE_PQ_64CENTROIDS(64);
        CASE_PQ_64CENTROIDS(66);
        CASE_PQ_64CENTROIDS(75);
        CASE_PQ_64CENTROIDS(85);
        CASE_PQ_64CENTROIDS(100);
        CASE_PQ_64CENTROIDS(105);
        CASE_PQ_64CENTROIDS(125);
        CASE_PQ_64CENTROIDS(128);
        CASE_PQ_64CENTROIDS(140);
        CASE_PQ_64CENTROIDS(160);
        CASE_PQ_64CENTROIDS(166);
        CASE_PQ_64CENTROIDS(170);
        CASE_PQ_64CENTROIDS(196);
        CASE_PQ_64CENTROIDS(240);
        CASE_PQ_64CENTROIDS(261);
        CASE_PQ_64CENTROIDS(342);
        CASE_PQ_64CENTROIDS(343);
        CASE_PQ_64CENTROIDS(320);
        CASE_PQ_64CENTROIDS(456);
        CASE_PQ_64CENTROIDS(1024);
        CASE_PQ_64CENTROIDS(1365);
        default:
        assert(false && "Unsupported mSubspaceNum for searchHeapSIMDLargeCentroids IVF path");
      }

    } else if (mCentroidsNum == 32) {
      switch (mSubspaceNum) {
        CASE_PQ_32CENTROIDS(8);
        CASE_PQ_32CENTROIDS(10);
        CASE_PQ_32CENTROIDS(16);
        CASE_PQ_32CENTROIDS(21);
        CASE_PQ_32CENTROIDS(22);
        CASE_PQ_32CENTROIDS(25);
        CASE_PQ_32CENTROIDS(32);
        CASE_PQ_32CENTROIDS(33);
        CASE_PQ_32CENTROIDS(42);
        CASE_PQ_32CENTROIDS(50);
        CASE_PQ_32CENTROIDS(70);
        CASE_PQ_32CENTROIDS(83);
        CASE_PQ_32CENTROIDS(85);
        CASE_PQ_32CENTROIDS(130);
        CASE_PQ_32CENTROIDS(160);
        CASE_PQ_32CENTROIDS(228);
        CASE_PQ_32CENTROIDS(682);
        default:
          assert(false && "Unsupported mSubspaceNum for searchHeapSIMD IVF path");
      }
    }
  }


}
