#include "PQ.h"
#include <unordered_map>
#include <array>
#include <optional>
#include "utils/prefetch.h"
#include <omp.h>
#include <numa.h>

#ifndef PARTITIONED_PQ_H
#define PARTITIONED_PQ_H

class PartitionedPQ{
public:
    using CodewordType = PQ::CodewordType;

    int mSubspaceNum;
    int mCentroidsNum;
    int mClustersNum;
    int minimumClusterSize;
    struct Cluster{
        PQ pq;
        std::vector<int> toOriginalID;
    };
    RowMatrixXf centroids;
    Eigen::VectorXf centroidsNorm2;
    std::vector<Cluster> clusters;

    #ifdef MY_TEST_QUANTIZED_LUT_DISTRIBUTION
    inline static std::array<int, 256> distDistribution = {};
    #endif

    void clusterVectors(RowMatrixXf &XTrain, const int numClusters, 
        std::vector<RowMatrixXf>& clusteredVectors, std::vector<RowVectorXf>&Centroids, float sampleRate=1);

    void parseMethodString(std::string methodString);
    void train(RowMatrixXf &XTrain, bool verbose = false, const float alpha = 1.0);
    void trainIVF(RowMatrixXf &XTrain, bool verbose = false, const float alpha = 1.0, const int nlist=512);
    //void encode(const RowMatrixXf &XTrain);
    //RowMatrixXf decode();

    void truncateCandidates(LabelDistVecF& ret);

    template<typename LUTDType=float>
    LabelDistVecF search(const RowMatrixXf &XTest, const RowMatrixXf &dataset, const int k, int topNCluster=-1, bool simd=false, float upSampleRate=1.,
         bool verbose=false, const RowMatrixXf *rawVectors=nullptr, int refine=0, const int nprobe=64, const int searchThread = 1);

    PQ::Distribution getFloatDistDistribution(const RowMatrixXf &XTest);

    using PPQDistribution = std::vector<std::vector<PQ::SingleDistribution>>; // [Query][partition]

    PPQDistribution getPartitionDistribution(const RowMatrixXf &XTest, const int topNClusters);

    float load(const std::string& filepath);
    void save(const std::string& filepath);

    void getCentroidsDistribution(const RowMatrixXf &XTest, int topNCluster) {

        // std::vector<double> top16, top32, top64, batch64;
        // top16.resize(mSubspaceNum);top32.resize(mSubspaceNum);top64.resize(mSubspaceNum);batch64.resize(mSubspaceNum);
        
        
        // for (int qIdx = 0; qIdx < XTest.rows(); qIdx++) {
        //     const RowVectorXf &curTest = XTest.row(qIdx);

        //     std::partial_sort(
        //         clusters.begin(), clusters.begin() + topNCluster, clusters.end(),
        //         [&curTest](const Cluster &lhs, const Cluster &rhs) {
        //             return (curTest - lhs.centroid).squaredNorm() < (curTest - rhs.centroid).squaredNorm();
        //         }
        //     );
        //         for (int cIdx = 0; cIdx < topNCluster; cIdx++) {
        //             auto result = this->clusters[cIdx].pq.computeCentroidDistribution();
        //             for(int sIdx = 0; sIdx < mSubspaceNum; ++sIdx){
        //                 // std::sort(result.ratios[sIdx].begin(), result.ratios[sIdx].end(), std::greater<double>());
        //                 double _top16 = std::accumulate(result.ratios[sIdx].begin(), result.ratios[sIdx].begin()+16, 0.0);
        //                 double _top32 = std::accumulate(result.ratios[sIdx].begin(), result.ratios[sIdx].begin()+32, 0.0);
        //                 double _top64 = std::accumulate(result.ratios[sIdx].begin(), result.ratios[sIdx].begin()+64, 0.0);
        //                 top16[sIdx] += _top16;
        //                 top32[sIdx] += _top32;
        //                 top64[sIdx] += _top64;
        //                 batch64[sIdx] += result.batchRatios[sIdx];
        //             }
        //         }
        // }

        // for(int sIdx = 0; sIdx < mSubspaceNum; ++sIdx){
        //     top16[sIdx] /= (topNCluster *  XTest.rows());
        //     top32[sIdx] /= (topNCluster *  XTest.rows());
        //     top64[sIdx] /= (topNCluster *  XTest.rows());
        //     batch64[sIdx] /= (topNCluster *  XTest.rows());

        //     printf("For Subspace %d, top16=%lf, top32=%lf, top64=%lf, batchTop64=%lf\n", 
        //         sIdx, top16[sIdx], top32[sIdx], top64[sIdx], batch64[sIdx]);
        // }
        assert(false && "Wait for updating partial sort");
    }

};

// This code is from https://github.com/luoxiao23333/HNSWPQ/blob/master/hnswlib/space_l2.h
static float
L2SqrSIMD16ExtAVX(const float *pVect1, const float *pVect2, const size_t qty) {

    alignas(32) float TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_storeu_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    const size_t done = qty16 << 4;
    for (size_t i = done; i < qty; ++i) {
        float diff = pVect1[i - done] - pVect2[i - done]; // 注意：此时 pVect1/pVect2 已经推进到 pEnd
        res += diff * diff;
    }
    return res;
}

// In search, we only get the id in partitionedPQ, and we should only obtain id in original vectors when calculating accuracy
// Not refine at search
// template<typename LUTDType>
// LabelDistVecF PartitionedPQ::search(const RowMatrixXf &XTest, const RowMatrixXf &dataset,
//     const int k, int topNCluster, bool simd, float upSampleRate, bool verbose,
//     const RowMatrixXf *rawVectors /* = nullptr */, int refine /* = 0 */,
//     const int nprobe /* = 64 */) {


//     if (topNCluster == -1) {
//         topNCluster = this->clusters.size();
//     }

//     #ifdef MY_TEST_QUANTIZED_LUT_DISTRIBUTION
//     #endif
//     assert(topNCluster <= clusters.size() && "topNCluster must be <= clusters.size()");

//     LabelDistVecF ret;
//     ret.labels.resize(XTest.rows()*k);
//     ret.distances.resize(XTest.rows()*k);
//     DataQuantizer quantizer;

//     for (int qIdx = 0; qIdx < XTest.rows(); qIdx++) {
//         const RowVectorXf &curTest = XTest.row(qIdx);
//         const float qNorm2 = curTest.squaredNorm();

//         Eigen::VectorXf dots = this->centroids * curTest.transpose(); 
//         Eigen::VectorXf centroidsDist = centroidsNorm2.array() + qNorm2 - 2.0f * dots.array(); 

//         std::vector<int> cIdxs(clusters.size());
//         std::iota(cIdxs.begin(), cIdxs.end(), 0);

//         std::partial_sort(
//             cIdxs.begin(), cIdxs.begin() + topNCluster, cIdxs.end(),
//             [&centroidsDist](const int &lhs, const int &rhs) {
//                 return centroidsDist(lhs) < centroidsDist(rhs);
//             }
//         );

//         struct LabelDistance {
//             int   label;
//             float distance;
//             LabelDistance() = default;
//         };

//         int upperK = static_cast<int>(upSampleRate * k);
//         // upperK = std::max(upperK, int(std::ceil(refine / float(topNCluster))));
//         upperK = std::max(upperK, refine);

//         for (int i = 0; i < topNCluster; i++) {
//             const int cIdx = cIdxs[i];
//             clusters[cIdx].pq.CreateLUT(curTest);
//         }

//         quantizer.trainQuick(clusters[cIdxs[0]].pq.lut, mSubspaceNum);
        
//         static thread_local PQ::TopKHeap topKHeap;
//         topKHeap.reset(refine);
//         int curNprobe = nprobe;
//         for (int i = 0; i < topNCluster; i++) {
//             const int cIdx = cIdxs[i];
//             int codebookRows = clusters[cIdx].pq.codebookRow;
//             int realK = std::min(upperK, static_cast<int>(codebookRows));
//             if (realK == 0) {
//                 continue;
//             }

//             LabelDistVecF topKResults;
//             assert(codebookRows >= minimumClusterSize);

//             if (true || (codebookRows > realK && codebookRows > minimumClusterSize)) {
//                 clusters[cIdx].pq.searchOneIVF(
//                     curTest, realK, curNprobe, simd, verbose, 
//                     topKHeap, cIdx, quantizer
//                 );
//                 curNprobe /= 3;
//                 curNprobe = std::max(4, curNprobe);
//                 #ifdef MY_TEST_QUANTIZED_LUT_DISTRIBUTION
//                 if (cIdx == topNCluster - 1) {
//                     for (uint8_t dist : topKResults.distances) {
//                         std::cout << (dist + 0) << std::endl;
//                         this->distDistribution[dist]++;
//                     }
//                 }
//                 #else
//                 #endif
//             } else {
//                 // 小 codebook：直接全扫 + float 距离
//                 std::vector<int> labels(realK);
//                 std::iota(labels.begin(), labels.end(), 0);
//             }

//         }



//         int oIDs[400];
//         for(int i = 0; i < refine; ++i) {
//             auto &curID = topKHeap.ids[i];
//             oIDs[i] = clusters[curID.clusterID].toOriginalID[curID.internalID];
//         }

//         // ========= 仅在此处插入“可选 refine 复排”，其余不动 =========
//         if (rawVectors != nullptr && refine > k) {

//             const auto rawRols = rawVectors->rows();
//             const auto dim = rawVectors->cols();
//             std::vector<float> exactDists(refine);
//             std::unordered_map<int, int> idMap;
//             std::vector<LabelDistance> allResults(refine);

//             for (int i = 0; i < refine; ++i) {
//                 // 1) prefetch future target(s)
//                 //    PFD=4 比较常见；如果 dim 很大且 L2/L3 miss 多，可以试 6~10
//                 constexpr int PFD = 6;

//                 if (i + PFD < refine) {
//                     const int lab_pf = oIDs[i + PFD];
//                     if ((unsigned)lab_pf < (unsigned)rawRols) {
//                         const float* b_pf =
//                             rawVectors->data() + (size_t)lab_pf * dim;
//                         // 只拉“头部”cache line 通常最值
//                         prefetch_vec_head(b_pf, dim);
//                     }
//                 }

//                 // （可选）如果 ids 基本无重复且很乱，可以再 prefetch 更近的一个
//                 // 这样能在小 refine/高 miss 时更稳，但也可能浪费带宽
//                 if (i + 1 < refine) {
//                     const int lab_pf1 = oIDs[i + 1];
//                     if ((unsigned)lab_pf1 < (unsigned)rawRols) {
//                         const float* b_pf1 =
//                             rawVectors->data() + (size_t)lab_pf1 * dim;
//                         _mm_prefetch(reinterpret_cast<const char*>(b_pf1), PREFETCH_HINT);
//                     }
//                 }

//                 // 2) actual compute
//                 const int lab = oIDs[i];
//                 if ((unsigned)lab < (unsigned)rawRols) {
//                     const float* b = rawVectors->data() + (size_t)lab * dim;
//                     const float exactDist = L2SqrSIMD16ExtAVX(curTest.data(), b, dim);
//                     allResults[i].distance = exactDist;
//                 } else {
//                     allResults[i].distance = std::numeric_limits<float>::infinity();
//                 }
//                 allResults[i].label = lab;
//             }


//             // 再次对前 R 做 partial_sort 取最终 top-k
            
//             // not full sort, so recall@R for R<k is not correct
//             std::nth_element(
//                 allResults.begin(),
//                 allResults.begin() + k,
//                 allResults.begin() + refine,
//                 [](const LabelDistance &a, const LabelDistance &b) {
//                     return a.distance < b.distance;
//                 }
//             );

//             auto *curLabel = ret.labels.data()+qIdx*k;
//             for(int i =0; i<k;++i) {
//                 (*curLabel) = allResults[i].label;
//                 curLabel++;
//             }
//             // copy dist here
//         } else {
//             // 保持你原有路径：直接按 distance 取前 k
//             std::copy_n(oIDs, k, ret.labels.data()+qIdx*k);
//         }
//         // ========= 可选 refine 结束 =========
//     }

//     return ret;
// }




// 你已有的 template
template<typename LUTDType>
LabelDistVecF PartitionedPQ::search(const RowMatrixXf &XTest, const RowMatrixXf &dataset,
    const int k, int topNCluster, bool simd, float upSampleRate, bool verbose,
    const RowMatrixXf *rawVectors /* = nullptr */, int refine /* = 0 */,
    const int nprobe /* = 64 */,
    const int searchThread /* = 1 */) {

    if (topNCluster == -1) topNCluster = (int)this->clusters.size();
    assert(topNCluster <= (int)clusters.size() && "topNCluster must be <= clusters.size()");

    LabelDistVecF ret;
    ret.labels.resize((size_t)XTest.rows() * (size_t)k);
    ret.distances.resize((size_t)XTest.rows() * (size_t)k);

    const int nq  = (int)XTest.rows();
    const int dim = (int)XTest.cols();

    // ====== 每个 query 一个 core：OpenMP 并行 ======
    #pragma omp parallel num_threads(searchThread)
    {
        // ---- 每线程私有对象（避免 shared race）----
        DataQuantizer quantizer_local;

        // 如果你确实需要 heap 的 thread_local，那保持不动也行
        // 这里仍然用你原来的 thread_local
        static thread_local PQ::TopKHeap topKHeap;

        // query 的 cluster index 缓冲
        std::vector<int> cIdxs;
        cIdxs.reserve(clusters.size());

        // refine 用的 id 缓冲
        std::vector<int> oIDs;
        oIDs.resize(std::max(refine, k));

        // 结果缓冲（refine rerank 用）
        struct LabelDistance { int label; float distance; };
        std::vector<LabelDistance> allResults;
        if (rawVectors != nullptr && refine > k) allResults.resize(refine);

        #pragma omp for schedule(static)
        for (int qIdx = 0; qIdx < nq; qIdx++) {

            const RowVectorXf &curTest = XTest.row(qIdx);
            const float qNorm2 = curTest.squaredNorm();

            Eigen::VectorXf dots = this->centroids * curTest.transpose();
            Eigen::VectorXf centroidsDist = centroidsNorm2.array() + qNorm2 - 2.0f * dots.array();

            // --- topNCluster selection ---
            if(clusters.size() != 1) {
                cIdxs.assign(clusters.size(), 0);
                std::iota(cIdxs.begin(), cIdxs.end(), 0);
                std::partial_sort(
                    cIdxs.begin(), cIdxs.begin() + topNCluster, cIdxs.end(),
                    [&centroidsDist](const int &lhs, const int &rhs) {
                        return centroidsDist(lhs) < centroidsDist(rhs);
                    }
                );
            } else {
                cIdxs.push_back(0);
            }

            int upperK = (int)(upSampleRate * k);
            upperK = std::max(upperK, refine);
            if (upperK <= 0) upperK = k;

            clusters[cIdxs[0]].pq.CreateLUT(curTest);
            quantizer_local.trainQuick(clusters[cIdxs[0]].pq.lut, mSubspaceNum);
            topKHeap.reset(refine);

            int curNprobe = nprobe;
            for (int i = 0; i < topNCluster; i++) {
                const int cIdx = cIdxs[i];
                const int codebookRows = clusters[cIdx].pq.codebookRow;
                const int realK = std::min(upperK, codebookRows);
                if (realK <= 0) continue;

                if(i) {
                    clusters[cIdx].pq.CreateLUT(curTest);
                }

                // 这里用新接口：传入该 cluster 对应的 LUT（线程私有）
                clusters[cIdx].pq.searchOneIVF(
                    curTest, realK, curNprobe, simd, verbose,
                    topKHeap, cIdx, quantizer_local
                );

                curNprobe = std::max(4, int(float(curNprobe) / 1.5));
            }

            // --- 4) heap ids -> original ids ---
            const int R = std::max(refine, k);
            for (int i = 0; i < R; ++i) {
                auto &curID = topKHeap.ids[i];
                oIDs[i] = clusters[curID.clusterID].toOriginalID[curID.internalID];
            }

            // --- 5) 可选 raw refine rerank ---
            if (rawVectors != nullptr && refine > k) {
                const int rawRows = (int)rawVectors->rows();
                const int rawDim  = (int)rawVectors->cols();

                for (int i = 0; i < refine; ++i) {
                    constexpr int PFD = 6;
                    if (i + PFD < refine) {
                        const int lab_pf = oIDs[i + PFD];
                        if ((unsigned)lab_pf < (unsigned)rawRows) {
                            const float* b_pf = rawVectors->data() + (size_t)lab_pf * rawDim;
                            prefetch_vec_head(b_pf, rawDim);
                        }
                    }
                    if (i + 1 < refine) {
                        const int lab_pf1 = oIDs[i + 1];
                        if ((unsigned)lab_pf1 < (unsigned)rawRows) {
                            const float* b_pf1 = rawVectors->data() + (size_t)lab_pf1 * rawDim;
                            _mm_prefetch(reinterpret_cast<const char*>(b_pf1), PREFETCH_HINT);
                        }
                    }

                    const int lab = oIDs[i];
                    float dist = std::numeric_limits<float>::infinity();
                    if ((unsigned)lab < (unsigned)rawRows) {
                        const float* b = rawVectors->data() + (size_t)lab * rawDim;
                        dist = L2SqrSIMD16ExtAVX(curTest.data(), b, rawDim);
                    }
                    allResults[i].label = lab;
                    allResults[i].distance = dist;
                }

                std::nth_element(
                    allResults.begin(),
                    allResults.begin() + k,
                    allResults.begin() + refine,
                    [](const LabelDistance &a, const LabelDistance &b) {
                        return a.distance < b.distance;
                    }
                );

                int *outLabels = ret.labels.data() + (size_t)qIdx * (size_t)k;
                for (int i = 0; i < k; ++i) outLabels[i] = allResults[i].label;

            } else {
                std::copy_n(oIDs.data(), k, ret.labels.data() + (size_t)qIdx * (size_t)k);
            }

        }
    } // end omp parallel

    return ret;
}


#endif // PARTITIONED_PQ_H