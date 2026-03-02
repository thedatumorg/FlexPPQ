#ifndef IO_HPP_
#define IO_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <cerrno>

#include <sys/stat.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

#include "Types.hpp"

inline bool isFileExists(const std::string& fname) {
  struct stat buffer;
  return (stat(fname.c_str(), &buffer) == 0);
}

/* Input functions */
template <bool IgnoreFirstCol=false>
inline void readOriginalFromExternal(std::string filepath, Matrixf &dataset, int N, char delim=',') {
  std::ifstream infile;
  infile.open(filepath);

  std::string line, bit, temp;
  while (!infile.eof()) {
    std::vector<float> v(N);
    
    std::getline(infile, line);
    if (line.empty()) { break; }
    std::stringstream ss(line);

    int colcounter = 0;
    while (std::getline(ss, bit, delim)) {
      if (IgnoreFirstCol && colcounter == 0) {
        colcounter++;
        continue;
      }

      if (IgnoreFirstCol) {
        v[colcounter-1] = std::stof(bit);
      } else {
        v[colcounter] = std::stof(bit);
      }
      colcounter++;
      if ((IgnoreFirstCol && colcounter > (int)N) || (!IgnoreFirstCol && colcounter >= (int)N)) {
        break;
      }
    }
    dataset.push_back(v);
  }
  infile.close();
}
template <bool IgnoreFirstCol=false, typename T>
inline void readOriginalFromExternal(std::string filepath, Eigen::MatrixBase<T> &dataset, int N, char delim=',') {
  std::ifstream infile;
  infile.open(filepath);

  int rowcounter = 0;
  std::string line, bit, temp;
  while (!infile.eof()) {
    std::getline(infile, line);
    if (line.empty()) { break; }
    std::stringstream ss(line);

    int colcounter = 0;
    while (std::getline(ss, bit, delim)) {
      if (IgnoreFirstCol && colcounter == 0) {
        colcounter++;
        continue;
      }

      if (IgnoreFirstCol) {
        dataset(rowcounter, colcounter-1) = std::stof(bit);
      } else {
        dataset(rowcounter, colcounter) = std::stof(bit);
      }
      colcounter++;
      if ((IgnoreFirstCol && colcounter > (int)N) || (!IgnoreFirstCol && colcounter >= (int)N)) {
        break;
      }
    }

    rowcounter++;
  }
  infile.close();
}

inline void readFVecsFromExternal(std::string filepath, Matrixf &dataset, int N, int maxRow=-1) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }
  
  int rowCt = 0;
  int dimen;
  while (true) {
    if (fread(&dimen, sizeof(int), 1, infile) == 0) {
      break;
    }
    if (dimen != N) {
      std::cout << "N and actual dimension mismatch" << std::endl;
      std::cout << "actual dims is " << dimen << ", reading dim is " << N << std::endl;
      assert(false);
      return;
    }
    std::vector<float> v(dimen);
    if(fread(v.data(), sizeof(float), dimen, infile) == 0) {
      std::cout << "Error when reading" << std::endl;
    };
    
    dataset.push_back(v);

    rowCt++;
    if (maxRow != -1 && rowCt >= maxRow) {
      break;
    }
  }

  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}
template<typename T>
inline void readFVecsFromExternal(std::string filepath, Eigen::MatrixBase<T> &dataset, int N, int maxRow=-1) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }
  
  int rowCt = 0;
  int dimen;
  while (true) {
    if (fread(&dimen, sizeof(int), 1, infile) == 0) {
      break;
    }
    if (dimen != N) {
      std::cout << "N and actual dimension mismatch" << std::endl;
      std::cout << "actual dims is " << dimen << ", reading dim is " << N << std::endl;
      return;
    }
    std::vector<float> v(dimen);
    if(fread(v.data(), sizeof(float), dimen, infile) == 0) {
      std::cout << "Error when reading" << std::endl;
    };
    
    for (int i=0; i<dimen; i++) {
      dataset(rowCt, i) = v[i];
    }

    rowCt++;
    if (maxRow != -1 && rowCt >= maxRow) {
      break;
    }
  }

  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}

inline void readBVecsFromExternal(std::string filepath, Matrixf &dataset, int N, int maxRow=-1) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }

  int dimen;
  int row = 0;  
  while (true) {
    if (fread(&dimen, sizeof(int), 1, infile) == 0) {
      break;
    }
    if (dimen != N) {
      std::cout << "dimension and N doesn't match" << std::endl;
    }

    std::vector<unsigned char> temp(N);
    if (fread(temp.data(), sizeof(unsigned char), N, infile) == 0) {
      break;
    }
    
    std::vector<float> v(temp.begin(), temp.end());  // convert to float
    dataset.push_back(v);
    row++;
    if (maxRow != -1 && row >= maxRow) {
      break;
    }
  }

  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}
template<typename T>
inline void readBVecsFromExternal(std::string filepath, Eigen::MatrixBase<T> &dataset, int N, int maxRow=-1) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }

  int dimen;
  int row = 0;  
  while (true) {
    if (fread(&dimen, sizeof(int), 1, infile) == 0) {
      break;
    }
    if (dimen != N) {
      std::cout << "dimension and N doesn't match" << std::endl;
    }

    std::vector<unsigned char> v(N);
    if (fread(v.data(), sizeof(unsigned char), N, infile) == 0) {
      break;
    }

    for (int i=0; i<dimen; i++) {
      dataset(row, i) = v[i];
    }
    
    row++;
    if (maxRow != -1 && row >= maxRow) {
      break;
    }
  }

  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}

inline void readFromExternalBin(std::string filepath, Matrixf &dataset, int N, int maxRow=-1) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }

  int row_ct = 0;
  while (true) {
    std::vector<float> v(N);
    if (fread(v.data(), sizeof(float), N, infile) == 0) {
      break;
    }
    
    dataset.push_back(v);
    row_ct++;
    if (maxRow != -1 && row_ct >= maxRow) {
      break;
    }
  }

  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}
template<typename T>
inline void readFromExternalBin(std::string filepath, Eigen::MatrixBase<T> &dataset, int N, int maxRow=-1) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }

  int row_ct = 0;
  while (true) {
    std::vector<float> v(N);
    if (fread(v.data(), sizeof(float), N, infile) == 0) {
      break;
    }
    
    for (int i=0; i<N; i++) {
      dataset(row_ct, i) = v[i];
    }

    row_ct++;
    if (maxRow != -1 && row_ct >= maxRow) {
      break;
    }
  }

  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}

inline void readTOPNNExternal(std::string filepath, std::vector<std::vector<int>> &dataset, int N, char delim=',') {
  std::ifstream infile;
  infile.open(filepath);
  
  std::string line, bit, temp;
  while (!infile.eof()) {
    std::vector<int> v(N);
    
    std::getline(infile, line);
    if (line.empty()) { break; }
    std::stringstream ss(line);

    int colcounter = 0;
    while (std::getline(ss, bit, delim)) {
      v[colcounter] = std::stoi(bit);
      colcounter++;
    }

    dataset.push_back(v);
  }
  infile.close();
}

inline void readTOPNNExternalBin(std::string filepath, std::vector<std::vector<int>> &dataset, int N) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }

  while (true) {
    std::vector<int> v(N);
    if (fread(v.data(), sizeof(int), N, infile) == 0) {
      break;
    }

    dataset.push_back(v);
  }
  
  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}

inline void readIVecsFromExternal(std::string filepath, std::vector<std::vector<int>> &dataset, int N) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }
  
  int dimen;
  while (true) {
    if (fread(&dimen, sizeof(int), 1, infile) == 0) {
      break;
    }
    if (dimen != N) {
      std::cout << "N and actual dimension mismatch" << std::endl;
      std::cout << "actual dims is " << dimen << ", reading dim is " << N << std::endl;
      assert(false);
      return;
    }
    std::vector<int> v(dimen);
    if (fread(v.data(), sizeof(int), dimen, infile) == 0) {
      std::cout << "error when reading" << std::endl;
    };
    
    dataset.push_back(v);
  }

  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}

inline void readFromExternal(std::string filepath, bitvectors &bv, int cols, char delim=',') {
  std::ifstream infile;
  infile.open(filepath);
  
  std::string line, bit, temp;
  while (!infile.eof()) {
    bitv v(actualBitVLen(cols), 0); 

    std::getline(infile, line);
    if (line.empty()) { break; }
    std::stringstream ss(line);

    int counter = 0;
    uint64_t vTemp = 0;
    while (std::getline(ss, bit, delim) && counter < cols) {
      vTemp |= (uint64_t)std::stoi(bit);

      counter++;
      if (counter % 64 == 0) {
        v[(counter / 64)-1] = vTemp;
        
        vTemp = 0;
      } else {
        vTemp <<= 1;
      }
    }

    if (counter % 64 != 0) {
      v[counter / 64] = vTemp << (64-(counter % 64));
    }

    bv.push_back(v);
  }
  infile.close();
}

template<int IdxOffset=0>
inline void readClusterIndexExternal(std::string filepath, std::vector<int> &clusterIdx) {
  std::ifstream infile;
  infile.open(filepath);
  
  std::string line, bit;
  while (!infile.eof()) {
    std::getline(infile, line);
    if (line.empty()) { break; }
    int idx = std::stoi(line)-IdxOffset;
    
    clusterIdx.push_back(idx);
  }
}

inline void readClassificationInfoFromExternal(std::string filepath, std::vector<int> & classInfo) {
  std::ifstream infile;
  infile.open(filepath);

  std::string line, bit, temp;
  while (!infile.eof()) {
    std::getline(infile, line);
    if (line.empty()) { break; }
    std::stringstream ss(line);

    if(std::getline(ss, bit, ',')) {
      classInfo.push_back(std::stoi(bit));  // only read the first value per row
    }
  }
  infile.close();
}

inline void readBVecsFromExternalSample(std::string filepath, Matrixf &dataset, int N, int maxRow=-1, int batch=1000000) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }

  fseek(infile, 0L, SEEK_END);
  unsigned long long sz = (unsigned long long) ftell(infile);
  unsigned long long total_records = sz/(sizeof(int) + N * sizeof(unsigned char));
  fseek(infile, 0L, SEEK_SET);

  unsigned long long sampled_each_batch = maxRow / ceil((float)total_records/batch);
  unsigned long long sampled_needed_remaining = maxRow;

  int dimen;
  while (sampled_needed_remaining > 0) {
    int b_ct = 0;
    Matrixf tempDataset;
    while (b_ct < batch) {
      if (fread(&dimen, sizeof(int), 1, infile) == 0) {
        break;
      }
      if (dimen != N) {
        std::cout << "dimension and N doesn't match" << std::endl;
      }

      std::vector<unsigned char> temp(N);
      if (fread(temp.data(), sizeof(unsigned char), N, infile) == 0) {
        break;
      }
      std::vector<float> v(temp.begin(), temp.end());  // convert to float

      tempDataset.push_back(v);
      b_ct++;
    }
    
    unsigned long long row_thisbatch = sampled_each_batch < sampled_needed_remaining ? sampled_each_batch : sampled_needed_remaining;
    for (unsigned long long i=0; i<row_thisbatch; i++) {
      dataset.push_back(tempDataset[rand() % b_ct]);
    }

    sampled_needed_remaining -= row_thisbatch;
  }

  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}

inline void readFromExternalBinSample(std::string filepath, Matrixf &dataset, int N, int maxRow=-1, int batch = 100000) {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }

  fseek(infile, 0L, SEEK_END);
  unsigned long long sz = (unsigned long long) ftell(infile);
  unsigned long long total_records = sz/(N * sizeof(float));
  fseek(infile, 0L, SEEK_SET);

  int sampled_each_batch = maxRow / ceil((float)total_records/batch);
  int sampled_needed_remaining = maxRow;

  while (sampled_needed_remaining > 0) {
    int b_ct = 0;
    Matrixf tempDataset;
    while (b_ct < batch) {
      std::vector<float> v(N);
      if (fread(v.data(), sizeof(float), N, infile) == 0) {
        break;
      }

      tempDataset.push_back(v);
      b_ct++;
    }
    
    int row_thisbatch = sampled_each_batch < sampled_needed_remaining ? sampled_each_batch : sampled_needed_remaining;
    for (int i=0; i<row_thisbatch; i++) {
      dataset.push_back(tempDataset[rand() % b_ct]);
    }
    
    sampled_needed_remaining -= row_thisbatch;
  }

  if (fclose(infile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}

inline CentroidsPerSubsType loadCentroids(std::string filepath)  {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "[utils/IO.hpp] loadCentroids(): File not found" << std::endl;
    exit(1);
  }
  size_t dim;
  if (fread(&dim, sizeof(size_t), 1, infile) == 0) {
    std::cout << "[utils/IO.hpp] loadCentroids(): Error while reading file" << std::endl;
    exit(1);
  }
  CentroidsPerSubsType centroids(dim); 
  for (size_t i=0; i<dim; i++) {
    size_t row, col;
    if ((fread(&row, sizeof(size_t), 1, infile) == 0) || (fread(&col, sizeof(size_t), 1, infile) == 0)) {
      std::cout << "[utils/IO.hpp] loadCentroids(): Error while reading file" << std::endl;
      exit(1);
    }
    CentroidsMatType means(row, col);
    if (fread(means.data(), sizeof(float), row*col, infile) == 0) {
      std::cout << "[utils/IO.hpp] loadCentroids(): Error while reading file" << std::endl;
      exit(1);
    }
    centroids[i] = means;
  }
  
  if (fclose(infile)) {
    std::cout << "[utils/IO.hpp] loadCentroids(): Could not close data file" << std::endl;
    exit(1);
  }

  return centroids;
}

inline CentroidsPerSubsType loadCentroids(std::ifstream &in) {
    if (!in.is_open()) {
        std::cerr << "[utils/IO.hpp] loadCentroids(): Invalid input stream" << std::endl;
        exit(1);
    }

    size_t dim;
    in.read(reinterpret_cast<char*>(&dim), sizeof(size_t));
    if (!in) {
        std::cerr << "[utils/IO.hpp] loadCentroids(): Error while reading dimension" << std::endl;
        exit(1);
    }

    CentroidsPerSubsType centroids(dim);
    for (size_t i = 0; i < dim; i++) {
        size_t row, col;
        in.read(reinterpret_cast<char*>(&row), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&col), sizeof(size_t));
        if (!in) {
            std::cerr << "[utils/IO.hpp] loadCentroids(): Error while reading row/col size" << std::endl;
            exit(1);
        }

        CentroidsMatType means(row, col);
        in.read(reinterpret_cast<char*>(means.data()), row * col * sizeof(float));
        if (!in) {
            std::cerr << "[utils/IO.hpp] loadCentroids(): Error while reading matrix data" << std::endl;
            exit(1);
        }
        centroids[i] = means;
    }

    return centroids;
}


template<typename CodebookTypeT>
inline CodebookTypeT loadCodebook(std::string filepath)  {
  FILE *infile = fopen(filepath.c_str(), "rb");
  if (infile == NULL) {
    std::cout << "[utils/IO.hpp] loadCodebook(): File not found " << filepath << std::endl;
  }
  size_t row, col;
  if ((fread(&row, sizeof(size_t), 1, infile) == 0) || (fread(&col, sizeof(size_t), 1, infile) == 0)) {
    std::cout << "[utils/IO.hpp] loadCodebook(): Error while reading file " << filepath << std::endl;
  }
  CodebookTypeT codebook(row, col);
  if (fread(codebook.data(), sizeof(typename CodebookTypeT::Scalar), row*col, infile) == 0) {
    std::cout << "[utils/IO.hpp] loadCodebook(): Error while reading file " << filepath << std::endl;
  }
  
  if (fclose(infile)) {
    std::cout << "[utils/IO.hpp] loadCodebook(): Could not close data file " << filepath << std::endl;
  }

  return codebook;
}

template <typename CodebookTypeT>
inline CodebookTypeT loadCodebook(std::ifstream &in) {
    if (!in.is_open()) {
        std::cerr << "[utils/IO.hpp] loadCodebook(): Invalid input stream" << std::endl;
        exit(1);
    }

    size_t row, col;
    in.read(reinterpret_cast<char*>(&row), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&col), sizeof(size_t));
    if (!in) {
        std::cerr << "[utils/IO.hpp] loadCodebook(): Error while reading row/col size" << std::endl;
        exit(1);
    }

    CodebookTypeT codebook(row, col);
    in.read(reinterpret_cast<char*>(codebook.data()), row * col * sizeof(typename CodebookTypeT::Scalar));
    if (!in) {
        std::cerr << "[utils/IO.hpp] loadCodebook(): Error while reading matrix data" << std::endl;
        exit(1);
    }

    return codebook;
}


template<typename MatrixType>
inline MatrixType loadMatrix(std::ifstream& in)  {
  typename MatrixType::Index rows=0, cols=0;
  in.read((char*) (&rows),sizeof(typename MatrixType::Index));
  in.read((char*) (&cols),sizeof(typename MatrixType::Index));
  MatrixType matrix(rows, cols);
  in.read((char *) matrix.data() , rows*cols*sizeof(typename MatrixType::Scalar));
  return matrix;
}

inline RowMatrix<Eigen::scomplex> loadEigenVectors(std::ifstream &in) {
  RowMatrix<Eigen::scomplex> eigenVectors;
  if (in.is_open()) {
      int rows = 0;
      int cols = 0;
      in.read(reinterpret_cast<char*>(&rows), sizeof(int));
      in.read(reinterpret_cast<char*>(&cols), sizeof(int));

      eigenVectors.resize(rows, cols);

      in.read(reinterpret_cast<char*>(eigenVectors.data()), rows * cols * sizeof(Eigen::scomplex));
  } else {
      std::cerr << "File open failed for loadEigenVectors" << std::endl;
      exit(1);
    }

  return eigenVectors;
}

/* Output functions */
inline void writeCentroidsExternalBolt(std::string filepath, const CentroidsPerSubsType &centroids) {
  std::ofstream outfile;
  outfile.open(filepath);

  for (int dim=0; dim<(int)centroids.size(); dim++) {
    for (int ci=0; ci<centroids[dim].rows(); ci++) {
      for (int i=0; i<centroids[dim].cols(); i++) {
        outfile << centroids[dim](ci, i);
        if (i < centroids[dim].cols()-1) {
          outfile << ',';
        }
      }
      outfile << std::endl;
    }
  }

  outfile.close();
}

inline void writeClusterIndexExternal(std::string filepath, std::vector<int> &clusterIdx) {
  std::ofstream outfile;
  outfile.open(filepath);

  for (auto clus: clusterIdx) {
    outfile << clus << std::endl;
  }

  outfile.close();
}

inline void writeTOPNNExternalBin(std::string filepath, const Matrixi &topnn) {
  FILE *outfile = fopen(filepath.c_str(), "wb");
  if (outfile == NULL) {
    std::cout << "File not found" << std::endl;
    return;
  }
  for (auto row: topnn) {
    fwrite(row.data(), sizeof(int), row.size(), outfile);
  }

  if (fclose(outfile)) {
    std::cout << "Could not close data file" << std::endl;
  }
}

inline void writeCentroidsPerDimExternal(std::string filepath, CentroidsPerSubsType &centroidsPerDim) {
  std::ofstream outfile;
  outfile.open(filepath);

  for (auto &centroids: centroidsPerDim) {
    int dimen = centroids.cols();
    for (int row=0; row<(int)centroids.rows(); row++) {
      int col_ct = 0;
      for (int col=0; col<(int)centroids.cols(); col++) {
        outfile << centroids(row, col);
        col_ct++;
        if (col_ct < dimen) {
          outfile << ",";
        }
      }
      outfile << std::endl;
    }
  }

  outfile.close();
}

inline void writeCentroidsExternal(std::string filepath, Matrixf &centroids) {
  std::ofstream outfile;
  outfile.open(filepath);

  int dimen = centroids[0].size();
  for (auto row: centroids) {
    int col_ct = 0;
    for (auto col: row) {
      outfile << col;
      col_ct++;
      if (col_ct < dimen) {
        outfile << ",";
      }
    }
    outfile << std::endl;
  }

  outfile.close();
}

inline void writeCentroidsExternal(std::string filepath, const Eigen::MatrixXf &centroids) {
  std::ofstream outfile;
  outfile.open(filepath);

  int dimen = centroids.cols();
  for (int row=0; row<(int)centroids.rows(); row++) {
    int col_ct = 0;
    for (int col=0; col<(int)centroids.cols(); col++) {
      outfile << centroids(row, col);
      col_ct++;
      if (col_ct < dimen) {
        outfile << ",";
      }
    }
    outfile << std::endl;
  }

  outfile.close();
}

inline void writeToExternal(std::string filepath, const bitvectors &bv, const int N) {
  std::ofstream outfile;
  outfile.open(filepath);

  for (const bitv &row: bv) {
    for (int i=0; i<(int)row.size(); i++) {
      const int maxBin = ((i+1)*64 <= N) ? 64 : (N % 64);
      uint64_t val = row[i];
      for (int b=0; b<maxBin; b++) {
        int bin = (val >> (63-b)) & 1u;
        outfile << bin;
        if (b != maxBin-1) {
          outfile << ',';
        }
      }
      if (i != (int)row.size()-1) {
        outfile << ',';
      }
    }
    outfile << std::endl;
  }

  outfile.close();
}

inline void writeKNNResults(std::string filepath, const std::vector<std::vector<IdxDistPairFloat>> &results) {
  std::ofstream outfile;
  outfile.open(filepath);
  for (size_t i=0; i<results.size(); i++) {
    for (size_t j=0; j<results[i].size(); j++) {
      outfile << results[i][j].idx;
      if (j != results[i].size()-1) {
        outfile << ',';
      }
    }
    outfile << std::endl;
  }
  outfile.close();
}
inline void writeKNNResults(std::string filepath, const LabelDistVecF &results, const size_t nrows) {
  const int k = (results.labels.size() / nrows);
  std::ofstream outfile;
  outfile.open(filepath);
  for (size_t i=0; i<nrows; i++) {
    for (size_t j=0; j<k; j++) {
      outfile << results.labels[i * k + j];
      if (j != k-1) {
        outfile << ',';
      }
    }
    outfile << std::endl;
  }
  outfile.close();
}

inline void saveCentroids(const CentroidsPerSubsType &centroids, std::string filepath)  {
  FILE *outfile = fopen(filepath.c_str(), "wb");
  if (outfile == NULL) {
    std::cout << "[utils/IO.hpp] saveCentroids(): File Open Failed" << std::endl;
    exit(1);
  }
  size_t dim = centroids.size();

  if(fwrite(&dim, sizeof(size_t), 1, outfile) == 0) {
    std::cout << "[utils/IO.hpp] saveCentroids(): Write Error" << std::endl;
    exit(1);
  }
  for (const CentroidsMatType &cmat: centroids) {
    size_t row = cmat.rows(), col = cmat.cols();
    fwrite(&row, sizeof(size_t), 1, outfile);
    fwrite(&col, sizeof(size_t), 1, outfile);
    fwrite(cmat.data(), sizeof(float), row*col, outfile);
  }

  if(ferror(outfile)) {
    std::cout << "[utils/IO.hpp] saveCentroids(): " << std::strerror(errno) << std::endl;
    exit(1);
  }
  
  if (fclose(outfile)) {
    std::cout << "[utils/IO.hpp] saveCentroids(): Could not close data file" << std::endl;
    exit(1);
  }
}

inline void saveCentroids(const CentroidsPerSubsType &centroids, std::ofstream &out) {
  if (!out) {
    std::cerr << "[utils/IO.hpp] saveCentroids(): Invalid output stream" << std::endl;
    exit(1);
  }

  size_t dim = centroids.size();

  // Write the dimension
  out.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
  if (!out) {
    std::cerr << "[utils/IO.hpp] saveCentroids(): Write Error (dim)" << std::endl;
    exit(1);
  }

  // Write each centroid matrix
  for (const CentroidsMatType &cmat : centroids) {
    size_t row = cmat.rows(), col = cmat.cols();
    out.write(reinterpret_cast<const char*>(&row), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&col), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(cmat.data()), sizeof(float) * row * col);
    if (!out) {
      std::cerr << "[utils/IO.hpp] saveCentroids(): Write Error (matrix data)" << std::endl;
      exit(1);
    }
  }
}


template<typename CodebookTypeT>
inline void saveCodebook(const CodebookTypeT &codebook, std::string filepath)  {
  FILE *outfile = fopen(filepath.c_str(), "wb");
  if (outfile == NULL) {
    std::cout << "[utils/IO.hpp] saveCodebook(): File not found " << filepath << std::endl;
    return;
  }
  
  size_t row = codebook.rows(), col = codebook.cols();
  fwrite(&row, sizeof(size_t), 1, outfile);
  fwrite(&col, sizeof(size_t), 1, outfile);
  fwrite(codebook.data(), sizeof(typename CodebookTypeT::Scalar), row * col, outfile);
  
  if (fclose(outfile)) {
    std::cout << "[utils/IO.hpp] saveCodebook(): Could not close file " << filepath << std::endl;
    exit(1);
  }
}

template <typename CodebookTypeT>
inline void saveCodebook(const CodebookTypeT &codebook, std::ofstream &out) {
  if (!out) {
    std::cerr << "[utils/IO.hpp] saveCodebook(): Invalid output stream" << std::endl;
    return;
  }

  size_t row = codebook.rows(), col = codebook.cols();
  
  // Write row and column sizes
  out.write(reinterpret_cast<const char*>(&row), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(&col), sizeof(size_t));

  // Write codebook data
  out.write(reinterpret_cast<const char*>(codebook.data()), sizeof(typename CodebookTypeT::Scalar) * row * col);
  if (!out) {
    std::cerr << "[utils/IO.hpp] saveCodebook(): Write Error" << std::endl;
    exit(1);
  }
}


template<typename MatrixType>
inline void saveMatrix(const MatrixType &matrix, std::ofstream& out)  {
  typename MatrixType::Index rows=matrix.rows(), cols=matrix.cols();
  out.write((char*) (&rows), sizeof(typename MatrixType::Index));
  out.write((char*) (&cols), sizeof(typename MatrixType::Index));
  out.write((char*) matrix.data(), rows*cols*sizeof(typename MatrixType::Scalar) );
}

inline void saveEigenVectors(const RowMatrix<Eigen::scomplex>& eigenVectors, std::ofstream& out) {
  if (out.is_open()) {
      int rows = static_cast<int>(eigenVectors.rows());
      int cols = static_cast<int>(eigenVectors.cols());
      out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
      out.write(reinterpret_cast<const char*>(&cols), sizeof(int));

      out.write(reinterpret_cast<const char*>(eigenVectors.data()), rows * cols * sizeof(Eigen::scomplex));
  } else {
      std::cerr << "[utils/IO.hpp] saveEigenVectors(): Could not open file " << std::endl;
      exit(1);
  }
}

template <typename VectorType>
inline void saveEigenVector(const VectorType& v, std::ofstream& out) {
    static_assert(VectorType::ColsAtCompileTime == 1 || VectorType::RowsAtCompileTime == 1,
                  "saveEigenVector: VectorType must be a vector");
    static_assert(std::is_trivially_copyable<typename VectorType::Scalar>::value,
                  "saveEigenVector: Scalar must be trivially copyable");

    int n = static_cast<int>(v.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof(int));
    out.write(reinterpret_cast<const char*>(v.data()),
              static_cast<std::streamsize>(n * sizeof(typename VectorType::Scalar)));
}

template <typename VectorType>
inline void loadEigenVector(VectorType& v, std::ifstream& in) {
    static_assert(VectorType::ColsAtCompileTime == 1 || VectorType::RowsAtCompileTime == 1,
                  "loadEigenVector: VectorType must be a vector");
    static_assert(std::is_trivially_copyable<typename VectorType::Scalar>::value,
                  "loadEigenVector: Scalar must be trivially copyable");

    int n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(int));
    assert(n >= 0);

    v.resize(n);
    in.read(reinterpret_cast<char*>(v.data()),
            static_cast<std::streamsize>(n * sizeof(typename VectorType::Scalar)));
}

template <typename Iterable>
inline void saveIterable(const Iterable& dataList, const int len, std::ofstream& out) {
    if (out.is_open()) {
        out.write(reinterpret_cast<const char*>(&len), sizeof(int));

        // for contiguous data layout
        if constexpr (std::is_same<typename std::iterator_traits<decltype(std::begin(dataList))>::iterator_category,
                                     std::random_access_iterator_tag>::value) {
            out.write(reinterpret_cast<const char*>(dataList.data()), len * sizeof(typename Iterable::value_type));
        } else {
            for (const auto& element : dataList) {
                out.write(reinterpret_cast<const char*>(&element), sizeof(element));
            }
        }
    } else {
        std::cerr << "[utils/IO.hpp] saveIterable(): Could not open file " << std::endl;
        exit(1);
    }
}

template <typename DType>
inline void saveOneData(const DType &data, std::ofstream& out) {
  out.write(reinterpret_cast<const char*>(&data), sizeof(DType));
}

template <>
inline void saveOneData<std::string>(const std::string &data, std::ofstream& out) {
  size_t length = data.size();
  out.write(reinterpret_cast<const char*>(&length), sizeof(size_t));
  out.write(data.data(), length);
}

template <typename DType>
inline void loadOneData(DType &data, std::ifstream& in) {
  in.read(reinterpret_cast<char*>(&data), sizeof(DType));
}

template <>
inline void loadOneData<std::string>(std::string &data, std::ifstream& in) {
    size_t length;
    in.read(reinterpret_cast<char*>(&length), sizeof(size_t));
    data.resize(length);
    in.read(&data[0], length);
}

template <typename Iterable>
inline void loadIterable(Iterable& dataList, std::ifstream& in) {
    if (in.is_open()) {
        int len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(int));

        if constexpr (std::is_same<typename std::iterator_traits<decltype(std::begin(dataList))>::iterator_category,
                                     std::random_access_iterator_tag>::value) {
            dataList.resize(len);
            in.read(reinterpret_cast<char*>(dataList.data()), len * sizeof(typename Iterable::value_type));
        } else {
            using ValueType = typename Iterable::value_type;
            dataList.clear();
            for (int i = 0; i < len; ++i) {
                ValueType element;
                in.read(reinterpret_cast<char*>(&element), sizeof(ValueType));
                dataList.insert(dataList.end(), element); 
            }
        }
    } else {
        std::cerr << "[utils/IO.hpp] loadIterable(): Could not open file " << std::endl;
        exit(1);
    }
}
#endif