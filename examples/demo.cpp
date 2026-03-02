
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>

#include <getopt.h>

#include <sys/sysinfo.h>
#include <sys/stat.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

// #define MY_TEST_QUANTIZED_LUT_DISTRIBUTION
#include "PartitionedPQ.h"
#include "utils/TimingUtils.hpp"
#include "utils/Experiment.hpp"
#include "utils/IO.hpp"
#include "utils/MetricLogger.hpp"

int main(int argc, char **argv) {
  std::vector<ArgsParse::opt> long_options {
    {"dataset", 's', ""},
    {"queries", 's', ""},
    {"file-format-ori", 's', "fvecs"},
    {"save", 's', ""}, // file path of centroids
    {"groundtruth", 's', ""},
    {"groundtruth-format", 's', "ascii"},
    {"result", 's', ""},
    {"dim", 'i', "1"}, // dim of each data vector
    {"dataset-size", 'i', "0"},
    {"queries-size", 'i', "0"},
    {"gt-dim", 'i', "100"},
    {"k", 'i', "100"}, // search for topk
    {"method", 's', "PQ(8,16)"}, // PQ(k,m) -> k: bits per Subspace, m: Subspace Length
    {"refine", 'i', "200"},
    {"search-topn", 'i', "-1"}, // only search top n cluster, -1 means search all clusters
    {"quant-bits", 'i', "4"}, // only support 1, 2 or 4 -> uint8, uint16, float
    {"simd", 'i', "0"}, // 0 for disable simd, 1 fot enable simd
    {"config-id", 's', "0"}, //
    {"metric-log", 's', ""},
    {"repeat", 'i', "1"}, // repeat to get accurate latency
    {"trial", 'i', "1"}, // trial to get accurate latency
    {"up-sample", 's', "1"},
    {"r-at-r", 'i', "0"},
    {"alpha", 's', "1.0"},
    {"nprobe", 'i', "64"},
    {"nlist", 'i', "512"},
    {"search_thread", 'i', "1"},
    {"centroid_distribution", 'i', "0"}
  };
  ArgsParse args = ArgsParse(argc, argv, long_options, "HELP");
  args.printArgs();

  // check if dataset and queries exist
  if (!isFileExists(args["dataset"]) || !isFileExists(args["queries"])) {
    std::cerr << "Dataset or queries file doesn't exists" << std::endl;
    return 1;
  }

  PartitionedPQ pq;
  pq.parseMethodString(args["method"]);
  const float alpha = std::stof(args["alpha"]);

  std::cout << "Preprocessing steps..\n" << std::endl;

  
  int dimPadding = 0;
  if (args.at<int>("dim") % pq.mSubspaceNum != 0) {
    std::cout << "padding cols, because the col of dataset can NOT be divisible by #subspace" << std::endl;
    int subvectorlen = args.at<int>("dim") / pq.mSubspaceNum;
    subvectorlen += (args.at<int>("dim") % pq.mSubspaceNum > 0) ? 1 : 0;
    dimPadding = (subvectorlen * pq.mSubspaceNum) - args.at<int>("dim");
  }
  RowMatrixXf dataset; 

  auto readDataset = [&args, &dataset, &dimPadding](){
    std::cout << "Read dataset" << std::endl;
    dataset = RowMatrixXf::Zero(args.at<int>("dataset-size"), args.at<int>("dim") + dimPadding);
    if (args["file-format-ori"] == "ascii") {
      readOriginalFromExternal<true>(args["dataset"], dataset, args.at<int>("dim"), ',');
    } else if (args["file-format-ori"] == "fvecs") {
      readFVecsFromExternal(args["dataset"], dataset, args.at<int>("dim"), args.at<int>("dataset-size"));
    } else if (args["file-format-ori"] == "bvecs") {
      readBVecsFromExternal(args["dataset"], dataset, args.at<int>("dim"), args.at<int>("dataset-size"));
    } else if (args["file-format-ori"] == "bin") {
      readFromExternalBin(args["dataset"], dataset, args.at<int>("dim"), args.at<int>("dataset-size"));
    }
    if(dataset.rows() != args.at<int>("dataset-size")) {
      printf("dataset size mismatch! readding %d, but param is %d\n", dataset.rows(), args.at<int>("dataset-size"));
      assert(false);
    }
  };
  double trainTime = 0;
  float imbalance = 0.;
  {

    std::cout << "Training & encoding phase" << std::endl;
    if (args["save"] != "" && isFileExists(args["save"])) {
      std::cout << "Reading saved centroids from " << args["save"] << std::endl;
      imbalance = pq.load(args["save"]);
    }

    START_TIMING(PQ_TRAINING);
    auto trainStart = std::chrono::steady_clock::now();
    if (args["save"] == "" || !isFileExists(args["save"])) {
      readDataset();
      std::cout << "Training the centroids" << std::endl;
      pq.trainIVF(dataset, true, alpha, args.at<int>("nlist"));
    }
    auto trainEnd = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = trainEnd - trainStart;
    trainTime = elapsed_seconds.count();
    END_TIMING(PQ_TRAINING, "== Training time: ");


    if (args["save"] != "" && !isFileExists(args["save"])) {
      std::cout << "Saving centroids to " << args["save"] << std::endl;
      pq.save(args["save"]);
    }
  }


  {
    RowMatrixXf queries = RowMatrixXf::Zero(args.at<int>("queries-size"), args.at<int>("dim") + dimPadding);

    std::cout << "Read queries" << pq.clusters.size() << std::endl;
    if (args["file-format-ori"] == "ascii") {
      readOriginalFromExternal<true>(args["queries"], queries, args.at<int>("dim"), ',');
    } else if (args["file-format-ori"] == "fvecs") {
      readFVecsFromExternal(args["queries"], queries, args.at<int>("dim"), args.at<int>("queries-size"));
    } else if (args["file-format-ori"] == "bvecs") {
      readBVecsFromExternal(args["queries"], queries, args.at<int>("dim"), args.at<int>("queries-size"));
    } else if (args["file-format-ori"] == "bin") {
      readFromExternalBin(args["queries"], queries, args.at<int>("dim"), args.at<int>("queries-size"));
    }

    std::vector<std::vector<int>> topnn;
    if (args["groundtruth"] != "") {
      std::cout << "Read groundtruth (search top " << args.at<int>("k") << ")" << std::endl;
      if (args["groundtruth-format"] == "ascii") {
        readTOPNNExternal(args["groundtruth"], topnn, args.at<int>("gt-dim"), ',');
      } else if (args["groundtruth-format"] == "ivecs") {
        readIVecsFromExternal(args["groundtruth"], topnn, args.at<int>("gt-dim"));
      } else if (args["groundtruth-format"] == "bin") {
        readTOPNNExternalBin(args["groundtruth"], topnn, args.at<int>("gt-dim"));
      }
    }

    if(args.at<int>("centroid_distribution") != 0) {
      pq.getCentroidsDistribution(queries, args.at<int>("search-topn"));
      exit(0);
    }

    std::cout << "Querying phase" << std::endl;
    LabelDistVecF answers;      

    int refine = args.at<int>("refine");
    if(refine > args.at<int>("k")){
      readDataset();
    }

    double queryTime = 0.;
    for (int t=0;t < args.at<int>("trial");t++) {
      double curTime=1e20;
      for (int i=0;i<args.at<int>("repeat");i++) {                                                                                                                         
        // int searchK = refine >= args.at<int>("k") ? refine : args.at<int>("k");
        int searchK = args.at<int>("k");
        bool simd = args.at<int>("simd") == 0? false:true;
        float upSampleRate = std::stof(args["up-sample"]);
        answers.distances.clear();
        answers.labels.clear();

        auto start = std::chrono::high_resolution_clock::now();
        if(args.at<int>("quant-bits") == 4){
          answers = pq.search<float>(queries, dataset, searchK, args.at<int>("search-topn"), simd, upSampleRate, true, &dataset, refine, args.at<int>("nprobe"), args.at<int>("search_thread")); 
        } else if(args.at<int>("quant-bits") == 2){
          answers = pq.search<uint16_t>(queries, dataset, searchK, args.at<int>("search-topn"), simd, upSampleRate, true, &dataset, refine, args.at<int>("nprobe"), args.at<int>("search_thread")); 
        } else if(args.at<int>("quant-bits") == 1){
          std::cout << "Search in uint8" << std::endl;
          answers = pq.search<uint8_t>(queries, dataset, searchK, args.at<int>("search-topn"), simd, upSampleRate, true, &dataset, refine, args.at<int>("nprobe"), args.at<int>("search_thread")); 
        } else {
          assert(false && "quantization only support 4B, 2B or 1B");
        }                   
        auto end = std::chrono::high_resolution_clock::now();                                                                  

        std::chrono::duration<double> duration = end - start;
        curTime = std::min(curTime, duration.count());
      }
      queryTime += (curTime / args.at<int>("trial"));
    }
    if (args["groundtruth"] != "") {
      // measure accuracy
      MetricLogger logger(args["metric-log"]);

      std::cout << "\trecallk@k" << args.at<int>("k") <<": " << getRecallAtR(answers.labels, topnn, args.at<int>("k"), args.at<int>("k"), bool(args.at<int>("r-at-r"))) << std::endl;
      logger.addLog(std::string("Recallk@k"), getRecallAtR(answers.labels, topnn, args.at<int>("k"), args.at<int>("k"), bool(args.at<int>("r-at-r"))));
      logger.addLog("k", args.at<int>("k"));

      logger.addLog("Latency", queryTime);
      logger.addLog("Throughput(Q/sec)", queries.rows()/queryTime);
      logger.addLog("TrainSec", trainTime);
      logger.addLog("Dataset", args["config-id"]);
      logger.addLog("NumCentroids", pq.mCentroidsNum);
      logger.addLog("NumSubspaces", pq.mSubspaceNum);
      logger.addLog("NumClusters", pq.mClustersNum);
      logger.addLog("Refine", args.at<int>("refine"));
      logger.addLog("TopNClusters", args.at<int>("search-topn"));
      logger.addLog("NLists", args.at<int>("nlist"));
      logger.addLog("NProbe", args.at<int>("nprobe"));
      logger.addLog("SearchThreads", args.at<int>("search_thread"));
      #ifdef MY_TEST_QUANTIZED_LUT_DISTRIBUTION
      std::string distributions;
      for(int distribution : pq.distDistribution) {
        distributions += std::to_string(distribution) + "-";
      }
      assert(!distributions.empty());
      distributions.pop_back();
      logger.addLog("Distributions", distributions);
      #endif
      logger.writeLog();
      

    }
  }

  return 0;
}
