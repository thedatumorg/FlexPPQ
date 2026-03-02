#include "PartitionedPQ.h"
#include <armadillo>

#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>
#include <random>

#include <armadillo>
#include <vector>

#include <algorithm>
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <stdexcept>
#include <chrono>
#include "utils/IO.hpp"

static inline std::vector<int> sampleRowIndices(int n, int k, uint64_t seed) {
    k = std::min(k, n);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(k);
    return idx;
}

// static inline std::vector<int> sampleRowIndices(int n, int k, uint64_t seed) {
//     k = std::min(k, n);
//     std::mt19937_64 rng(seed);

//     std::vector<int> out;
//     out.reserve(k);

//     // 定义一个虚拟整数迭代器 [0, n)
//     struct IntIter {
//         int v;
//         IntIter(int v) : v(v) {}
//         int operator*() const { return v; }
//         IntIter& operator++() { ++v; return *this; }
//         bool operator!=(const IntIter& other) const { return v != other.v; }
//     };

//     std::sample(IntIter(0), IntIter(n), std::back_inserter(out), k, rng);
//     return out;
// }

void PartitionedPQ::clusterVectors(RowMatrixXf &XTrain, 
                    const int numClusters, 
                    std::vector<RowMatrixXf>& clusteredVectors, 
                    std::vector<RowVectorXf>& centroids,
                    float sampleRate) 
{
    const int numPoints = XTrain.rows();  
    const int dim = XTrain.cols();       

    // int nSample = static_cast<int>(std::ceil(numPoints * sampleRate));
    // nSample = std::max(nSample, numClusters);
    // std::vector<int> sampleIdx = sampleRowIndices(numPoints, nSample, 1111);
    // RowMatrixXf XSample(nSample, dim);

    // #pragma omp parallel for schedule(static)
    // for (int i = 0; i < nSample; ++i) XSample.row(i) = XTrain.row(sampleIdx[i]);

    // arma::fmat armaXTrain(XSample.data(), dim, numPoints, false, false);
    arma::fmat armaXTrain(XTrain.data(), dim, numPoints, false, false);
    arma::fmat armaCentroids;
    auto start = std::chrono::steady_clock::now();
    const int maxit = 50;
    bool success = arma::kmeans(armaCentroids, armaXTrain, numClusters, arma::random_subset, maxit, false);
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    printf("KMeans in preclustering is %lf\n", elapsed);
    if (!success) {
        throw std::runtime_error("K-means clustering failed!");
    }

    centroids.clear();
    centroids.reserve(numClusters);
    for (int i = 0; i < numClusters; ++i) {
        centroids.push_back(Eigen::Map<RowVectorXf>(armaCentroids.colptr(i), dim).eval());
    }

    std::vector<int> clusterAssignments(numPoints, -1);
    clusteredVectors.clear();
    clusteredVectors.resize(numClusters);

    std::vector<int> clusterSizes(numClusters, 0);
    for (int i = 0; i < numPoints; ++i) {
        Eigen::VectorXf point = XTrain.row(i);
        float minDist = std::numeric_limits<float>::max();
        int bestCluster = -1;

        for (int j = 0; j < numClusters; ++j) {
            Eigen::VectorXf centroid = centroids[j];
            float dist = (point - centroid).squaredNorm();
            if (dist < minDist) {
                minDist = dist;
                bestCluster = j;
            }
        }

        clusterAssignments[i] = bestCluster;
        clusterSizes[bestCluster]++;
    }

    // RowMatrixXf C(numClusters, dim);
    // for (int j = 0; j < numClusters; ++j) C.row(j) = centroids[j];

    // // 预存范数
    // Eigen::VectorXf xn2(numPoints), cn2(numClusters);
    // #pragma omp parallel for schedule(static)
    // for (int i = 0; i < numPoints; ++i) xn2[i] = XTrain.row(i).squaredNorm();
    // #pragma omp parallel for schedule(static)
    // for (int j = 0; j < numClusters; ++j) cn2[j] = C.row(j).squaredNorm();

    // // 关键一步：G = X * C^T（用 Eigen/BLAS 多线程加速）
    // RowMatrixXf G = XTrain * C.transpose();

    // // 取 argmin
    // #pragma omp parallel
    // {
    //     std::vector<int> local(numClusters, 0);
    //     #pragma omp for schedule(static)
    //     for (int i = 0; i < numPoints; ++i) {
    //         float best = std::numeric_limits<float>::max();
    //         int bestJ = 0;
    //         for (int j = 0; j < numClusters; ++j) {
    //             float d = xn2[i] + cn2[j] - 2.0f * G(i, j);
    //             if (d < best) { best = d; bestJ = j; }
    //         }
    //         clusterAssignments[i] = bestJ;
    //         local[bestJ] += 1;
    //     }
    //     #pragma omp critical
    //     for (int j = 0; j < numClusters; ++j) clusterSizes[j] += local[j];
    // }

    for (int i = 0; i < numClusters; ++i) {
        clusteredVectors[i].resize(clusterSizes[i], dim);
        this->clusters[i].toOriginalID.resize(clusterSizes[i]);
    }

    std::vector<int> currentIndex(numClusters, 0);
    for (int i = 0; i < numPoints; ++i) {
        int cluster = clusterAssignments[i];
        clusteredVectors[cluster].row(currentIndex[cluster]) = XTrain.row(i);
        this->clusters[cluster].toOriginalID[currentIndex[cluster]++] = i;
    }

    // Sometime clustering is unbalanced, so we pad float_max, which will have the maximum distance with query.
    for (int i = 0; i < clusteredVectors.size(); ++i) {
        if(clusteredVectors[i].rows() < minimumClusterSize) {
            printf("Padding FLOAT_MAX to cluster %d, cause it only has %d rows\n", i, clusteredVectors[i].rows());

            for(int curIndex = clusteredVectors[i].rows(); curIndex < minimumClusterSize; ++curIndex) {
                this->clusters[i].toOriginalID.push_back(-10);
            }

            int padRows = minimumClusterSize - clusteredVectors[i].rows();
            RowMatrixXf extended(minimumClusterSize, clusteredVectors[i].cols());
            extended.topRows(clusteredVectors[i].rows()) = clusteredVectors[i];
            extended.bottomRows(padRows) = RowMatrixXf::Constant(padRows, clusteredVectors[i].cols(), std::numeric_limits<float>::max());
            clusteredVectors[i] = extended;   
        }
    }

}

void PartitionedPQ::train(RowMatrixXf &XTrain, bool verbose, const float alpha) {

    std::vector<RowMatrixXf> clusteredVectors;
    std::vector<RowVectorXf> centroids;
    minimumClusterSize = std::max(100, mCentroidsNum);
    this->clusters.resize(mClustersNum);

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    this->clusterVectors(XTrain, mClustersNum, clusteredVectors, centroids);
    auto end = high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    printf("pre clustering: %lf\n", elapsed);

    for(int i=0; i<clusteredVectors.size(); i++) {
        start = high_resolution_clock::now();
        this->clusters[i].pq.mCentroidsNum = this->mCentroidsNum;
        this->clusters[i].pq.mSubspaceNum = this-> mSubspaceNum;
        this->clusters[i].pq.train(clusteredVectors[i], verbose);
        this->clusters[i].pq.encode(clusteredVectors[i], this->clusters[i].toOriginalID, alpha);
        end = high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        printf("%dth clustering: %lf\n", i, elapsed);
    }

    this->centroids = RowMatrixXf(centroids.size(), centroids[0].size());
    for(int i = 0; i < centroids.size(); ++i) {
        this->centroids.row(i) = centroids[i]; 
    }
    this->centroidsNorm2 = this->centroids.rowwise().squaredNorm();
}

void PartitionedPQ::trainIVF(RowMatrixXf &XTrain, bool verbose, const float alpha, const int nlist) {
    std::vector<RowMatrixXf> clusteredVectors;
    std::vector<RowVectorXf> centroids;
    minimumClusterSize = std::max(100, mCentroidsNum);
    this->clusters.resize(mClustersNum);

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    this->clusterVectors(XTrain, mClustersNum, clusteredVectors, centroids);
    auto end = high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    printf("pre clustering: %lf\n", elapsed);

    for(int i=0; i<clusteredVectors.size(); i++) {
        start = high_resolution_clock::now();
        this->clusters[i].pq.mCentroidsNum = this->mCentroidsNum;
        this->clusters[i].pq.mSubspaceNum = this-> mSubspaceNum;
        this->clusters[i].pq.trainIVF(clusteredVectors[i], nlist, verbose);
        this->clusters[i].pq.encodeIVF(clusteredVectors[i]);
        end = high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        printf("%dth clustering: %lf\n", i, elapsed);
    }

    this->centroids = RowMatrixXf(centroids.size(), centroids[0].size());
    for(int i = 0; i < centroids.size(); ++i) {
        this->centroids.row(i) = centroids[i]; 
    }
    this->centroidsNorm2 = this->centroids.rowwise().squaredNorm();
}

void PartitionedPQ::parseMethodString(std::string methodString) {

  if (methodString.rfind("PQ", 0) == 0) {
    int bitPerCentroid;
    if (std::sscanf(methodString.c_str(), "PQ(%d,%d,%d)", &mClustersNum, &bitPerCentroid, &mSubspaceNum) == 3) {
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

void PartitionedPQ::truncateCandidates(LabelDistVecF& ret) {
    const size_t n = ret.distances.size();

    if (n <= 100) return;

    if (ret.distances[99] != 0) {
        ret.labels.resize(100);
        ret.distances.resize(100);
        return;
    }

    // Binary search for first non-zero element after index 100
    auto begin = ret.distances.begin() + 100;
    auto end = ret.distances.end();

    // Find first distance != 0
    auto it = std::find_if(begin, end, [](uint8_t v) { return v != 0; });

    size_t cutoff = std::distance(ret.distances.begin(), it);
    ret.labels.resize(cutoff);
    ret.distances.resize(cutoff);
}


float PartitionedPQ::load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);

    loadOneData(mClustersNum, in);
    //this->clusters.resize(mClustersNum);

    for (int cIdx=0; cIdx < mClustersNum; ++cIdx) {
        Cluster curCluster;

        loadOneData(curCluster.pq.mCentroidsNum, in);
        loadOneData(curCluster.pq.mSubspaceNum, in);
        loadOneData(curCluster.pq.mSubsLen, in);
        loadOneData(curCluster.pq.codebookRow, in);
        
        mCentroidsNum = curCluster.pq.mCentroidsNum;
        mSubspaceNum = curCluster.pq.mSubspaceNum;

        minimumClusterSize = std::max(100, mCentroidsNum);
        curCluster.pq.mCentroidsPerSubs = loadCentroids(in);
        curCluster.pq.mCodebook = loadCodebook<decltype(curCluster.pq.mCodebook)>(in);
        curCluster.pq.prepareSmallCodebook();
        // curCluster.pq.buildGroups();
        // curCluster.pq.mSmallCodebook = loadCodebook<decltype(curCluster.pq.mSmallCodebook)>(in);
        loadIterable(curCluster.pq.perm_sub, in);
        loadIterable(curCluster.pq.inv_perm_sub_, in);
        loadIterable(curCluster.pq.pruneMarks, in);
        curCluster.pq.mIVF.load(in);

        // curCluster.pq.mCentroidsPerSubsCMajor.resize(curCluster.pq.mCentroidsPerSubs.size());
        // for (int i=0; i<(int)curCluster.pq.mCentroidsPerSubs.size(); i++) {
        //     curCluster.pq.mCentroidsPerSubsCMajor[i] = curCluster.pq.mCentroidsPerSubs[i];
        // }

        loadIterable(curCluster.toOriginalID, in);

        this->clusters.push_back(curCluster);
    }
    this->centroids = loadMatrix<RowMatrixXf>(in);
    loadEigenVector(this->centroidsNorm2, in);

    std::vector<size_t> cSize;
    for(const Cluster &cur : this->clusters){
        cSize.push_back(cur.toOriginalID.size());
    }

    auto [minIt, maxIt] = std::minmax_element(cSize.begin(), cSize.end());
    if (*minIt == 0) return std::numeric_limits<double>::infinity(); // 有空簇则返回∞
    return static_cast<double>(*maxIt) / static_cast<double>(*minIt);
    
}

void PartitionedPQ::save(const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);

    saveOneData(mClustersNum, out);
    for (int cIdx=0; cIdx < mClustersNum; ++cIdx) {
        const Cluster& curCluster = this->clusters[cIdx];

        // save the PQ
        saveOneData(curCluster.pq.mCentroidsNum, out);
        saveOneData(curCluster.pq.mSubspaceNum, out);
        saveOneData(curCluster.pq.mSubsLen, out);
        saveOneData(curCluster.pq.codebookRow, out);
        saveCentroids(curCluster.pq.mCentroidsPerSubs, out);
        saveCodebook(curCluster.pq.mCodebook, out);
        saveIterable(curCluster.pq.perm_sub, curCluster.pq.perm_sub.size(), out);
        saveIterable(curCluster.pq.inv_perm_sub_, curCluster.pq.inv_perm_sub_.size(), out);
        saveIterable(curCluster.pq.pruneMarks, curCluster.pq.pruneMarks.size(), out);
        curCluster.pq.mIVF.save(out);

        //saveCodebook(curCluster.pq.mSmallCodebook, out);

        saveIterable(curCluster.toOriginalID, curCluster.toOriginalID.size(), out);
        
    }
    saveMatrix(this->centroids, out);
    saveEigenVector(this->centroidsNorm2, out);
    
}

// void PQ::loadMetaData(const std::string &filepath) {
//   std::ifstream in(filepath, std::ios::binary);

//   loadOneData(this->mCentroidsNum, in);
//   loadOneData(this->mSubspaceNum, in);
//   loadOneData(this->mSubsLen, in);

//   std::cout << "Meta Data of this codebook: " << std::endl
//     << "#Centroids per Subspace: " << mCentroidsNum << std::endl
//     << "#Subspaces: " << mSubspaceNum << std::endl;

//   in.close();
// }

// void PQ::saveMetaData(const std::string &filepath) {
//   std::ofstream out(filepath, std::ios::binary);

//   saveOneData(this->mCentroidsNum, out);
//   saveOneData(this->mSubspaceNum, out);
//   saveOneData(this->mSubsLen, out);

//   out.close();
// }

PQ::Distribution PartitionedPQ::getFloatDistDistribution(const RowMatrixXf &XTest) {
    //assert(clusers.size() == 1);
    return clusters[0].pq.getFloatDistDistribution(XTest);
}


PartitionedPQ::PPQDistribution PartitionedPQ::getPartitionDistribution(const RowMatrixXf &XTest, const int topNClusters) {
    // PartitionedPQ::PPQDistribution distributions;
    // distributions.resize(XTest.rows());
    // for(int qIdx=0; qIdx<XTest.rows(); qIdx++) {
    //     const RowVectorXf& curTest = XTest.row(qIdx);
    //     distributions[qIdx].resize(topNClusters);
    //     std::partial_sort(clusters.begin(), clusters.begin()+topNClusters, clusters.end(), 
    //         [&curTest](const Cluster&lhs, const Cluster&rhs){
    //             return (curTest - lhs.centroid).squaredNorm() < (curTest - rhs.centroid).squaredNorm();
    //         }
    //     );
    //     for(int i=0;i<topNClusters;++i) {
    //         distributions[qIdx][i] = clusters[i].pq.getSingleDistribution(curTest);
    //     }
    // }

    // return distributions;
    assert(false && "Wait for updation");
    return {};
}


