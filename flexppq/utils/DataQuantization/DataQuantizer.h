#ifndef DATA_QUANTIZER
#define DATA_QUANTIZER

#include <algorithm> 
#include <cmath>   
#include <vector>  
#include <cstdint>  
#include <limits>   
#include <iostream>

#include "../Types.hpp"



class DataQuantizer {

    template<typename Derived>
    std::vector<float> colwiseMedian(const Eigen::MatrixBase<Derived>& data) {
        int cols = data.cols();
        int rows = data.rows();
        std::vector<float> medians(cols);

        std::vector<float> buffer(rows); 

        for (int col = 0; col < cols; ++col) {
            std::copy(data.col(col).data(),  data.col(col).data() + rows, buffer.begin());
            // more efficient with median approximation
            std::nth_element(buffer.begin(), buffer.begin() + rows / 20, buffer.end());
            medians[col] = buffer[rows / 2];
        }

        return medians;
    }


    template<typename LUTDType>
    ColMatrix<LUTDType> normalizeAndScale(const ColMatrix<LUTDType>& data_in, const int numSubspaces) {
        #ifdef DISTABLE_NORMALIZATION
        return data_in;
        #endif

        using std::sqrt;

        int N = data_in.rows();
        int D = data_in.cols();
        // if(data_in.colwise().mean().sum()*5 >= data_in.rowwise().mean().sum()){
        //     return data_in;
        // }

        #ifdef DATA_QUANTIZER_DEBUG
        printMatrix(data_in, "Before Normalize Original LUT");
        #endif

        ColMatrix<LUTDType> data = data_in;  

        Eigen::Matrix<LUTDType, Eigen::Dynamic, 1> min = data.colwise().minCoeff(); 
        // auto mediums = colwiseMedian(data);
        for(int i=0;i<data.rows();++i) {
            for(int j=0;j<data.cols();++j) {
                auto threshold = min(j)*numSubspaces;
                if(data(i, j) > threshold)
                    data(i, j) = threshold;
            }
        }


        #ifdef DATA_QUANTIZER_DEBUG
        printMatrix(data, "After Normalize Original LUT");
        #endif

        return data;
    }

    constexpr static int MAX_DATASET=100, MIN_DATASET=50;
public:
    float qmin, qmax;

    // Use keep% codebook to compute the distances, and set qmax to the max distance, qmin to the min distance
    void trainQuick(const LUTType& lut, const size_t subspaceNum) {

            int adjustedSpaceNum = subspaceNum;
            while(adjustedSpaceNum>=1 && (lut.col(adjustedSpaceNum-1).isZero(1e-18))) adjustedSpaceNum--;
            auto trimmedLUT = lut.leftCols(adjustedSpaceNum);

            qmin = trimmedLUT.minCoeff();
            qmax = 0.;
            auto min = trimmedLUT.colwise().minCoeff(); 
            const auto mean = trimmedLUT.colwise().mean();

            // auto medians = colwiseMedian(trimmedLUT);
            // for (int j = 0; j < trimmedLUT.cols(); ++j) {
            //     auto threshold = min(j)*adjustedSpaceNum;
            //     qmax += medians[j];
            // }

            // adjustedSpaceNum = std::min(32, adjustedSpaceNum);
            // for (int j = 0; j < trimmedLUT.cols(); ++j) {
            //     auto threshold = min(j)*adjustedSpaceNum;
            //     qmax += std::min(trimmedLUT.col(j).mean(), threshold);
            // }

            adjustedSpaceNum = std::min(32, adjustedSpaceNum);
            qmax += mean.array()
              .min(min.array() * float(adjustedSpaceNum))
              .sum();

            // auto max = trimmedLUT.colwise().maxCoeff(); 
            // for (int j = 0; j < trimmedLUT.cols(); ++j) {
            //     qmax += max(j);
            // }

            return;

    }

    template<typename CBType>
    void train(const LUTType& lut, const CBType& codebook, bool quickTrain=false, float keep = 0.001) {

        if(quickTrain) {
            int subspaceNum = codebook.cols();
            int adjustedSpaceNum = subspaceNum;
            while(adjustedSpaceNum>=1 && (lut.col(adjustedSpaceNum-1).isZero(1e-20))) adjustedSpaceNum--;
            auto trimmedLUT = lut.leftCols(adjustedSpaceNum);

            qmin = trimmedLUT.minCoeff();
            qmax = 0.;
            auto min = trimmedLUT.colwise().minCoeff(); 

            // auto medians = colwiseMedian(trimmedLUT);
            // for (int j = 0; j < trimmedLUT.cols(); ++j) {
            //     auto threshold = min(j)*adjustedSpaceNum;
            //     qmax += std::min(medians[j], threshold);
            // }
            // return;
            
            for (int j = 0; j < trimmedLUT.cols(); ++j) {
                auto threshold = min(j)*adjustedSpaceNum;
                qmax += std::min(trimmedLUT.col(j).mean(), threshold);
            }
            return;
        }
    }

    template<typename Matrix>
    void printMatrix(Matrix&m, const std::string& info) {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << info << std::endl;
        for(int i=0;i<m.rows();++i) {
            for(int j=0;j<m.cols();++j){
                std::cout << m(i, j) + 0 << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
    }

    // qx = round(x-qmin/(qmax-qmin)*(dtype_max-dtype_min)+dtype_min)
    // quant to TargetDType
    template<typename TargetDType=uint8_t>
    ColMatrix<TargetDType> quantize(LUTType& originalLut) {
        constexpr TargetDType dtype_min = std::numeric_limits<TargetDType>::min();
        constexpr TargetDType dtype_max = std::numeric_limits<TargetDType>::max();
        // Create an output LUT of the same dimensions as the input
        ColMatrix<TargetDType> quantizedLut(originalLut.rows(), originalLut.cols());

        assert(qmin != qmax && "if qmin == qmax, then do not need quantization");

        // Perform the quantization
        auto* quant = quantizedLut.data();
        auto* original = originalLut.data();
        int lutSize = originalLut.size();

        while(lutSize--) {
            if(*original == 0) {
                *quant = 0;
            } else {
                float ratio = ((*original) - qmin) / (qmax - qmin);
                ratio =  ratio < 1.0f ? ratio : 1.0f;
                *quant = static_cast<TargetDType>(std::round(ratio * (dtype_max-dtype_min)));
                *quant += dtype_min;
            }
            original++;
            quant++;
        }

        #ifdef DATA_QUANTIZER_DEBUG
        printMatrix(quantizedLut, "Quantized LUT");
        #endif

        return quantizedLut;
    }

    // qx = round(x-qmin/(qmax-qmin)*(dtype_max-dtype_min)+dtype_min)
    // x = (qx-dtype_min)/(dtype_max-dtype_min)*(qmax-qmin)+qmin
    template<typename TargetDType=uint8_t>
    std::vector<float> dequantize(const std::vector<TargetDType>& quantizedVec) {
        constexpr TargetDType dtype_min = std::numeric_limits<TargetDType>::min();
        constexpr TargetDType dtype_max = std::numeric_limits<TargetDType>::max();

        assert(qmin < qmax && "qmin must be less than qmax");

        float scale = (qmax - qmin) / (dtype_max - dtype_min);
        std::vector<float> dequantizedVec(quantizedVec.size());

        // x = (qx - qmin) * scale + qmin
        for (size_t i = 0; i < quantizedVec.size(); ++i) {
            dequantizedVec[i] = (quantizedVec[i] - dtype_min) * scale + qmin;
        }

        return dequantizedVec;
    }

};


#endif // !DATA_QUANTIZER