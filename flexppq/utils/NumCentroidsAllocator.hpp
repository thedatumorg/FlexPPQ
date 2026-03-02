#ifndef NUM_CENTROIDS_ALLOCATOR
#define NUM_CENTROIDS_ALLOCATOR

#include <iostream>
#include <vector>
#include <glpk.h>
#include <cassert>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

// Configuration structure
struct Configuration {
    int centroids;  // Number of centroids
    double latency; // Latency of this configuration
    static const Configuration NONE;
    bool operator== (const Configuration&RHS) {
        return centroids == RHS.centroids && latency == RHS.latency;
    }
    bool operator!= (const Configuration&RHS) {
        return !(*this == RHS);
    }
    static std::vector<Configuration> loadFromFile(const std::string& fileName) {
        std::vector<Configuration> configs;
        std::ifstream file(fileName);

        if (!file) {
            std::cerr << "Error: Unable to open file " << fileName << std::endl;
            return configs;  // Return empty vector if file cannot be opened
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            Configuration config;
            char comma;

            if (ss >> config.centroids >> comma >> config.latency) {
                if (comma != ',') {
                    assert((std::string("Error: Invalid format: ") + line).c_str());
                }
                configs.push_back(config);
            } else {
                assert((std::string("Error: Invalid format: ") + line).c_str());
            }
        }

        return configs;
    }
};

inline const std::vector<Configuration> defaultConfigs = std::vector<Configuration>{
    {16, 4.48}, {32, 7.1}, {64, 12.46}, {128, 21.54}
};

inline const Configuration Configuration::NONE = Configuration{-1, 0};

class NumCentroidsAllocator {
private:
    int numSubspaces; // Number of subspaces (m)
    int numConfigurations; // Number of shared configurations (n)
    std::vector<double> importanceWeights; // Importance weights (W)
    double latencyBudget; // Latency budget (L)
    std::vector<Configuration> configurations; // Shared configuration options
    std::vector<Configuration> allocation; // Selected configuration for each subspace

public:
    // Constructor
    NumCentroidsAllocator(int subspaces, const std::vector<double>& weights, double budget,
                              const std::vector<Configuration> config)
        : numSubspaces(subspaces), numConfigurations(config.size()), importanceWeights(weights),
          latencyBudget(budget), configurations(config), allocation(subspaces, Configuration::NONE) {
            double lower = numSubspaces*configurations[0].latency;
            double upper = numSubspaces*configurations[configurations.size()-1].latency;
            std::cout << "#Subspaces, Latency Assignment Range: " << numSubspaces << ", [" 
                << lower << ", " << upper << "]" << std::endl; 
            if(latencyBudget <= 1.1) {
                latencyBudget = (upper-lower)*latencyBudget + lower;
            }

          }

    enum class OPT {Regular = 0, Grouped = 1, Proportional = 2, UA = 3};

    void solve(const int opt) {
        switch(static_cast<OPT>(opt)) {
            case OPT::Regular: return solveRegular();
            case OPT::Grouped: return solveGrouped();
            case OPT::Proportional: return solveProportional();
            case OPT::UA: return uniformAllocation();
            default: assert(false && "Unsuporrted opt for NumCentroidsAllocator");
        }
    }

    // Solve the optimization problem
    void solveRegular() {

        std::cout << "Start Solving Optimization Problem" << std::endl;
        std::cout << "Latency Budget: " << latencyBudget << std::endl;
        std::cout << "Importance: [";
        for(auto w:importanceWeights){
            std::cout << w << ", ";
        }
        std::cout << "]\n";

        std::cout << "Real Latency Budget: " << latencyBudget << std::endl;

        glp_prob* problem = glp_create_prob();
        glp_set_prob_name(problem, "Product Quantization Optimization");
        glp_set_obj_dir(problem, GLP_MAX); // Maximization problem

        // Add constraints for subspaces and latency budget
        glp_add_rows(problem, numSubspaces + 1);
        for (int i = 1; i <= numSubspaces; ++i) {
            glp_set_row_bnds(problem, i, GLP_FX, 1.0, 1.0); // Each subspace selects exactly one configuration
        }
        glp_set_row_bnds(problem, numSubspaces + 1, GLP_UP, 0.0, latencyBudget); // Latency budget constraint

        // Add columns for decision variables
        // glp_add_cols(problem, numSubspaces * numConfigurations);
        // int columnIndex = 1;
        // for (int i = 0; i < numSubspaces; ++i) {
        //     for (int j = 0; j < numConfigurations; ++j) {
        //         glp_set_col_bnds(problem, columnIndex, GLP_DB, 0.0, 1.0); // y_ij in {0, 1}, double bounds
        //         glp_set_col_kind(problem, columnIndex, GLP_BV);           // Binary variable
        //         glp_set_obj_coef(problem, columnIndex, importanceWeights[i] * configurations[j].centroids); // Objective coefficient
        //         ++columnIndex;
        //     }
        // }
        glp_add_cols(problem, numSubspaces * numConfigurations);
        int columnIndex = 1;
        for (int i = 0; i < numSubspaces; ++i) {
            for (int j = 0; j < numConfigurations; ++j) {
                if (i != 0 && configurations[j].centroids > 128) {
                    glp_set_col_bnds(problem, columnIndex, GLP_FX, 0.0, 0.0);
                    glp_set_obj_coef(problem, columnIndex, 0.0);
                } else {
                    glp_set_col_bnds(problem, columnIndex, GLP_DB, 0.0, 1.0);
                    glp_set_obj_coef(problem, columnIndex, importanceWeights[i] * configurations[j].centroids);
                }
                glp_set_col_kind(problem, columnIndex, GLP_BV); 
                ++columnIndex;
            }
        }

        // Define the constraint matrix
        int numNonZeroElements = numSubspaces * numConfigurations + numSubspaces * numConfigurations;
        std::vector<int> rowIndices(numNonZeroElements + 1);
        std::vector<int> colIndices(numNonZeroElements + 1);
        std::vector<double> coefficients(numNonZeroElements + 1);
        int matrixIndex = 1;

        // Subspace constraints
        for (int i = 0; i < numSubspaces; ++i) {
            for (int j = 0; j < numConfigurations; ++j) {
                rowIndices[matrixIndex] = i + 1;
                colIndices[matrixIndex] = i * numConfigurations + j + 1;
                coefficients[matrixIndex] = 1.0;
                ++matrixIndex;
            }
        }

        // Latency constraint
        for (int i = 0; i < numSubspaces; ++i) {
            for (int j = 0; j < numConfigurations; ++j) {
                rowIndices[matrixIndex] = numSubspaces + 1;
                colIndices[matrixIndex] = i * numConfigurations + j + 1;
                coefficients[matrixIndex] = configurations[j].latency;
                ++matrixIndex;
            }
        }

        glp_load_matrix(problem, numNonZeroElements, rowIndices.data(), colIndices.data(), coefficients.data());

        int simplexResult = glp_simplex(problem, NULL);
        if (simplexResult != 0) {
            std::cerr << "Simplex failed with error code: " << simplexResult << std::endl;
            glp_delete_prob(problem);
            assert(false);
        }

        // Solve the integer programming problem
        glp_intopt(problem, NULL);

        // Retrieve results and store assignments
        for (int i = 0; i < numSubspaces; ++i) {
            for (int j = 0; j < numConfigurations; ++j) {
                if (glp_mip_col_val(problem, i * numConfigurations + j + 1) > 0.5) {
                    allocation[i] = configurations[j];
                }
            }
        }

        // Free GLPK resources
        glp_delete_prob(problem);

        for(int i = 0; i < numSubspaces; ++i) {
            assert(allocation[i] != Configuration::NONE && "Each subspaces must have one assignment");
        }
    }

    void solveGrouped() {
        std::cout << "Start Solving Optimization Problem (Grouped Version)" << std::endl;
        std::cout << "Latency Budget: " << latencyBudget << std::endl;
        std::cout << "Importance Weights: [";
        for(auto w : importanceWeights){
            std::cout << w << ", ";
        }
        std::cout << "]\n";
        double lower = numSubspaces*configurations[0].latency;
        double upper = numSubspaces*configurations[configurations.size()-1].latency;

        std::cout << "Real Latency Budget: " << latencyBudget << std::endl;

        // ======================================================
        // 第一阶段：对子空间进行分组（基于 importanceWeights）
        // ======================================================
        // 构造 (importance, subspace index) 对，便于排序
        std::vector<std::pair<double, int>> weightIndex;
        for (int i = 0; i < numSubspaces; ++i) {
            weightIndex.push_back(std::make_pair(importanceWeights[i], i));
        }
        std::sort(weightIndex.begin(), weightIndex.end(), [](const auto &a, const auto &b) {
            return a.first > b.first;
        });

        // 定义分组阈值：如果相邻子空间权重之差不超过该阈值，则归为同一组
        double threshold = (*std::max_element(importanceWeights.begin(), importanceWeights.end()) - *std::min_element(importanceWeights.begin(), importanceWeights.end())) * 0.25;

        // 定义局部结构体 Group 用于存储分组结果
        struct Group {
            std::vector<int> indices; // 组内子空间的下标（原始编号）
            double totalWeight;       // 组内所有子空间的权重之和
            int size;                 // 组内子空间个数
        };

        std::vector<Group> groups;
        Group currentGroup;
        // 先将排序后的第一个子空间放入第一组
        currentGroup.indices.push_back(weightIndex[0].second);
        currentGroup.totalWeight = weightIndex[0].first;
        int groupIndex = 0;
        for (size_t i = 1; i < weightIndex.size(); i++) {
            double diff = weightIndex[groupIndex].first - weightIndex[i].first;
            printf("%lf, %lf, %lf\n", threshold, weightIndex[groupIndex].first, weightIndex[i].first);
            if (diff <= threshold) {
                // 同一组：加入当前组
                currentGroup.indices.push_back(weightIndex[i].second);
                currentGroup.totalWeight += weightIndex[i].first;
            } else {
                // 不同组：保存当前组，然后新开一组
                currentGroup.size = currentGroup.indices.size();
                groups.push_back(currentGroup);
                currentGroup.indices.clear();
                currentGroup.totalWeight = 0.0;
                currentGroup.indices.push_back(weightIndex[i].second);
                currentGroup.totalWeight = weightIndex[i].first;
                groupIndex = i+1;
            }
        }
        currentGroup.size = currentGroup.indices.size();
        groups.push_back(currentGroup);

        int numGroups = groups.size();

        std::cout << "Grouping Results (" << numGroups << " groups):" << std::endl;
        for (int g = 0; g < numGroups; g++) {
            std::cout << "  Group " << g << ": ";
            for (int idx : groups[g].indices) {
                std::cout << idx << " (w=" << importanceWeights[idx] << ") ";
            }
            std::cout << " | TotalWeight=" << groups[g].totalWeight 
                      << ", Size=" << groups[g].size << std::endl;
        }

        // ======================================================
        // 第二阶段：对各组选择配置（规划问题）
        // ======================================================
        // 模型：对每个组 g 与配置 j，建立二进制变量 y_{g,j}
        // 目标：最大化 sum_{g,j} (group_totalWeight * configurations[j].centroids) * y_{g,j}
        // 约束：
        //    (1) 每个组必须选择一个配置： sum_j y_{g,j} = 1, for each group g.
        //    (2) 全局 latency 约束：
        //         sum_{g,j} (group_size * configurations[j].latency) * y_{g,j} <= latencyBudget.
        
        int numConfigs = numConfigurations;  // 已经在构造函数中设定： configurations.size()
        int numVars = numGroups * numConfigs;  // 每个组有 numConfigs 个变量

        // 创建 GLPK 问题
        glp_prob* problem = glp_create_prob();
        glp_set_prob_name(problem, "Grouped_Num_Centroids_Allocation");
        glp_set_obj_dir(problem, GLP_MAX);

        // 添加约束：共 (numGroups + 1) 行
        // 前 numGroups 行为每个组必须选择一个配置的约束，
        // 第 (numGroups+1) 行为全局 latency 约束
        glp_add_rows(problem, numGroups + 1);
        for (int g = 1; g <= numGroups; ++g) {
            std::string rowName = "group_" + std::to_string(g);
            glp_set_row_name(problem, g, rowName.c_str());
            glp_set_row_bnds(problem, g, GLP_FX, 1.0, 1.0); // 等式约束：恰选一个配置
        }
        int latencyRow = numGroups + 1;
        glp_set_row_name(problem, latencyRow, "latency");
        glp_set_row_bnds(problem, latencyRow, GLP_UP, 0.0, latencyBudget);

        // 添加变量：每个变量代表某组选择某配置，共 numVars 个变量
        glp_add_cols(problem, numVars);
        int colIndex = 1;
        for (int g = 0; g < numGroups; ++g) {
            for (int j = 0; j < numConfigs; ++j) {
                std::string colName = "y_" + std::to_string(g) + "_" + std::to_string(j);
                glp_set_col_name(problem, colIndex, colName.c_str());
                glp_set_col_bnds(problem, colIndex, GLP_DB, 0.0, 1.0); // 取值 0 或 1
                glp_set_col_kind(problem, colIndex, GLP_BV);
                // 目标函数系数：组总权重 * 配置 j 的 centroids
                double coef = groups[g].totalWeight * configurations[j].centroids;
                glp_set_obj_coef(problem, colIndex, coef);
                ++colIndex;
            }
        }

        // 构造约束矩阵
        // 注意：GLPK 矩阵下标从 1 开始
        // 非零元素个数：组约束每个变量系数为 1，共 numGroups*numConfigs 个；
        //          全局 latency 约束每个变量系数为 (group_size * config[j].latency)，共 numGroups*numConfigs 个；
        int totalNonZeros = numVars * 2;
        std::vector<int> ia(totalNonZeros + 1);
        std::vector<int> ja(totalNonZeros + 1);
        std::vector<double> ar(totalNonZeros + 1);
        int matrixIndex = 1;
        // 第一部分：组约束
        for (int g = 0; g < numGroups; ++g) {
            for (int j = 0; j < numConfigs; ++j) {
                ia[matrixIndex] = g + 1; // 第 g+1 行（组约束）
                ja[matrixIndex] = g * numConfigs + j + 1; // 对应变量
                ar[matrixIndex] = 1.0;
                ++matrixIndex;
            }
        }
        // 第二部分：全局 latency 约束
        for (int g = 0; g < numGroups; ++g) {
            for (int j = 0; j < numConfigs; ++j) {
                ia[matrixIndex] = latencyRow; // 全局 latency 行
                ja[matrixIndex] = g * numConfigs + j + 1;
                // 每个变量的 latency 消耗为：该组子空间个数 * 配置 j 的 latency
                ar[matrixIndex] = groups[g].size * configurations[j].latency;
                ++matrixIndex;
            }
        }
        glp_load_matrix(problem, totalNonZeros, ia.data(), ja.data(), ar.data());

        // 求解模型：先调用单纯形，再求整数解
        int simplexResult = glp_simplex(problem, NULL);
        if (simplexResult != 0) {
            std::cerr << "Simplex failed with error code: " << simplexResult << std::endl;
            glp_delete_prob(problem);
            assert(false);
        }
        glp_intopt(problem, NULL);

        // 读取求解结果：对每个组，确定选择的配置，然后将该配置分配给组内所有子空间
        for (int g = 0; g < numGroups; ++g) {
            int chosenConfig = -1;
            for (int j = 0; j < numConfigs; ++j) {
                int varIndex = g * numConfigs + j + 1;
                if (glp_mip_col_val(problem, varIndex) > 0.5) {
                    chosenConfig = j;
                    break;
                }
            }
            // 若该组中找到一个配置，则为组内所有子空间赋值
            if (chosenConfig != -1) {
                for (int subspaceIdx : groups[g].indices) {
                    allocation[subspaceIdx] = configurations[chosenConfig];
                }
            }
        }

        // 清理 GLPK 资源
        glp_delete_prob(problem);

        // 检查所有子空间是否都有分配
        for (int i = 0; i < numSubspaces; ++i) {
            assert(allocation[i] != Configuration::NONE && "Each subspace must have one assignment");
        }
    }

    void solveProportional() {
        std::cout << "Start Solving Proportional Optimization Problem" << std::endl;
        std::cout << "Latency Budget: " << latencyBudget << std::endl;
        std::cout << "Importance Weights: [";
        for (auto w : importanceWeights) {
            std::cout << w << ", ";
        }
        std::cout << "]\n";

        // 惩罚系数 lambda，用于在目标函数中惩罚比例偏差
        // double lambdaParam = configurations[configurations.size()-1].latency * 10; // 你可以根据需要调整
        // lambda = avg(config latency) * numSubspaces
        double lambdaParam = 0.;
        for(auto config : configurations) {
            lambdaParam += config.latency;
        }
        lambdaParam /= configurations.size();
        lambdaParam *= numSubspaces;

        // 记 m = numSubspaces, n = numConfigurations
        int m = numSubspaces;
        int n = numConfigurations;
        
        // 总决策变量数量：
        //  x_{ij} (m*n 个) + k (1 个) + epsilon_i (m 个)
        int numXVars = m * n;
        int idx_k = numXVars + 1;
        int numEpsilon = m;
        int totalVars = numXVars + 1 + numEpsilon; // m*n + 1 + m

        // 总约束数量：
        //  (1) 每个子空间一条赋值约束: m
        //  (2) 全局 latency 约束: 1
        //  (3) 对每个子空间比例约束：下界 + 上界各 m，总计 2*m
        // 总计: m + 1 + 2*m = 3*m + 1
        int totalRows = 3 * m + 1;

        // 创建 GLPK 问题
        glp_prob* problem = glp_create_prob();
        glp_set_prob_name(problem, "Proportional_Num_Centroids_Allocation");
        glp_set_obj_dir(problem, GLP_MAX); // 最大化目标

        // 添加所有行（约束）
        glp_add_rows(problem, totalRows);
        int row = 0;
        // (1) 每个子空间必须选择一个配置：sum_j x_{ij} = 1, 行 1~m
        for (int i = 0; i < m; ++i) {
            row = i + 1;
            std::string rowName = "assign_" + std::to_string(i);
            glp_set_row_name(problem, row, rowName.c_str());
            glp_set_row_bnds(problem, row, GLP_FX, 1.0, 1.0);
        }
        // (2) 全局 latency 约束：sum_{i,j} L_j * x_{ij} <= latencyBudget, 行 m+1
        row = m + 1;
        glp_set_row_name(problem, row, "latency");
        glp_set_row_bnds(problem, row, GLP_UP, 0.0, latencyBudget);
        
        // (3) 比例约束：对于每个子空间 i，要求
        //      k*w_i - epsilon_i <= sum_j c_j * x_{ij} <= k*w_i + epsilon_i.
        //    将其拆成两个约束：
        //    (a) 下界: sum_j c_j * x_{ij} - k*w_i + epsilon_i >= 0, 行 m+2 ~ m+1+m
        //    (b) 上界: sum_j c_j * x_{ij} - k*w_i - epsilon_i <= 0, 行 2*m+2 ~ 2*m+1+m
        for (int i = 0; i < m; ++i) {
            // 下界约束
            row = m + 1 + i + 1; // 行号从 m+2 开始
            std::string rowName = "prop_lower_" + std::to_string(i);
            glp_set_row_name(problem, row, rowName.c_str());
            // 形式： expression >= 0
            glp_set_row_bnds(problem, row, GLP_LO, 0.0, 0.0);
            
            // 上界约束
            row = 2 * m + 1 + i + 1; // 行号从 2*m+2 开始
            rowName = "prop_upper_" + std::to_string(i);
            glp_set_row_name(problem, row, rowName.c_str());
            // 形式： expression <= 0
            glp_set_row_bnds(problem, row, GLP_UP, 0.0, 0.0);
        }
        
        // 添加所有列（决策变量）
        glp_add_cols(problem, totalVars);
        int col = 0;
        // (a) x_{ij} 二进制变量，索引 1 ~ m*n
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                col = i * n + j + 1;
                std::string colName = "x_" + std::to_string(i) + "_" + std::to_string(j);
                glp_set_col_name(problem, col, colName.c_str());
                glp_set_col_bnds(problem, col, GLP_DB, 0.0, 1.0);
                glp_set_col_kind(problem, col, GLP_BV);
                // 目标函数系数：w_i * c_j
                double coef = importanceWeights[i] * configurations[j].centroids;
                glp_set_obj_coef(problem, col, coef);
            }
        }
        // (b) k 变量，索引 = m*n + 1，连续，非负
        col = numXVars + 1;
        glp_set_col_name(problem, col, "k");
        glp_set_col_bnds(problem, col, GLP_LO, 0.0, 0.0); // lower bound 0, 无上界
        glp_set_obj_coef(problem, col, 0.0); // k 不出现在目标函数

        // (c) epsilon_i 变量，索引 = m*n + 1 + i, 连续，非负，目标函数系数为 -lambda
        for (int i = 0; i < m; ++i) {
            col = numXVars + 2 + i; // Shift by one so that epsilon_0 gets index m*n+2
            std::string colName = "epsilon_" + std::to_string(i);
            glp_set_col_name(problem, col, colName.c_str());
            glp_set_col_bnds(problem, col, GLP_LO, 0.0, 0.0);
            glp_set_obj_coef(problem, col, -lambdaParam);
        }

        // 构造约束矩阵
        // 非零元总数计算：
        //   - Assignment约束: m*n 个
        //   - Latency约束: m*n 个
        //   - 对于每个子空间：
        //         Lower proportional: n (来自 x_{ij}) + 1 (来自 k) + 1 (来自 epsilon) = n+2
        //         Upper proportional: 同样 n+2
        //   总计: m*n + m*n + m*(n+2) + m*(n+2) = 4*m*n + 4*m = 4*m*(n+1)
        int totalNonZeros = 4 * m * (n + 1);
        std::vector<int> ia(totalNonZeros + 1);
        std::vector<int> ja(totalNonZeros + 1);
        std::vector<double> ar(totalNonZeros + 1);
        int matrixIndex = 1;
        
        // 1. Assignment约束：行 1 ~ m
        for (int i = 0; i < m; ++i) {
            int rowIndex = i + 1;
            for (int j = 0; j < n; ++j) {
                int varIndex = i * n + j + 1;
                ia[matrixIndex] = rowIndex;
                ja[matrixIndex] = varIndex;
                ar[matrixIndex] = 1.0;
                ++matrixIndex;
            }
        }
        
        // 2. Latency约束：行 m+1
        {
            int rowIndex = m + 1;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    int varIndex = i * n + j + 1;
                    ia[matrixIndex] = rowIndex;
                    ja[matrixIndex] = varIndex;
                    ar[matrixIndex] = configurations[j].latency;
                    ++matrixIndex;
                }
            }
        }
        
        // 3. 下界比例约束：对于每个子空间 i, 行 = m + i + 2
        for (int i = 0; i < m; ++i) {
            int rowIndex = m + i + 2;
            // x_{ij} 项：系数 = c_j (配置 j 的 centroids)
            for (int j = 0; j < n; ++j) {
                int varIndex = i * n + j + 1;
                ia[matrixIndex] = rowIndex;
                ja[matrixIndex] = varIndex;
                ar[matrixIndex] = configurations[j].centroids;
                ++matrixIndex;
            }
            // k 项：系数 = -w_i
            ia[matrixIndex] = rowIndex;
            ja[matrixIndex] = numXVars + 1; // k的变量索引
            ar[matrixIndex] = -importanceWeights[i];
            ++matrixIndex;
            // epsilon_i 项：系数 = +1
            ia[matrixIndex] = rowIndex;
            ja[matrixIndex] = numXVars + 2 + i;
            ar[matrixIndex] = 1.0;
            ++matrixIndex;
        }
        
        // 4. 上界比例约束：对于每个子空间 i, 行 = 2*m + i + 2
        for (int i = 0; i < m; ++i) {
            int rowIndex = 2 * m + i + 2;
            // x_{ij} 项：系数 = c_j
            for (int j = 0; j < n; ++j) {
                int varIndex = i * n + j + 1;
                ia[matrixIndex] = rowIndex;
                ja[matrixIndex] = varIndex;
                ar[matrixIndex] = configurations[j].centroids;
                ++matrixIndex;
            }
            // k 项：系数 = -w_i
            ia[matrixIndex] = rowIndex;
            ja[matrixIndex] = numXVars + 1;
            ar[matrixIndex] = -importanceWeights[i];
            ++matrixIndex;
            // epsilon_i 项：系数 = -1
            ia[matrixIndex] = rowIndex;
            ja[matrixIndex] = numXVars + 2 + i;
            ar[matrixIndex] = -1.0;
            ++matrixIndex;
        }
        
        // 加载矩阵数据到 GLPK 模型
        glp_load_matrix(problem, totalNonZeros, ia.data(), ja.data(), ar.data());
        
        // 求解模型：先用单纯形，再用整数规划求解
        int simplexResult = glp_simplex(problem, NULL);
        if (simplexResult != 0) {
            std::cerr << "Simplex failed with error code: " << simplexResult << std::endl;
            glp_delete_prob(problem);
            assert(false);
        }
        glp_intopt(problem, NULL);
        
        // 读取求解结果，并根据 x_{ij} 变量确定每个子空间选择的配置
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int varIndex = i * n + j + 1;
                if (glp_mip_col_val(problem, varIndex) > 0.5) {
                    allocation[i] = configurations[j];
                    break;
                }
            }
        }
        
        // 可选：输出求解得到的比例系数 k 与偏差 epsilon
        double k_val = glp_mip_col_val(problem, numXVars + 1);
        std::cout << "Optimal k = " << k_val << std::endl;
        for (int i = 0; i < m; ++i) {
            double eps_val = glp_mip_col_val(problem, numXVars + 1 + i);
            std::cout << "Subspace " << i << " epsilon = " << eps_val << std::endl;
        }
        
        // 清理 GLPK 资源
        glp_delete_prob(problem);
        
        // 检查每个子空间是否都有分配
        for (int i = 0; i < numSubspaces; ++i) {
            assert(allocation[i] != Configuration::NONE && "Each subspace must have one assignment");
        }
    }

    void uniformAllocation() {
        Configuration config(Configuration::NONE);
        for(auto curConfig : configurations) {
            if(curConfig.centroids == importanceWeights[0]) {
                config = curConfig;
            }
        }

        if(config == Configuration::NONE) {
            config = Configuration{static_cast<int>(importanceWeights[0]), 0};
        }

        for(int i=0;i<numSubspaces; ++i) {
            allocation.push_back(config);
        }
    }

    // Get the selected configurations
    std::vector<Configuration> getAllocation() const {
        return allocation;
    }

    std::vector<int> getNumCentroids() const {
        std::vector<int> numCentroids;
        for(auto config : allocation) {
            numCentroids.push_back(config.centroids);
        }
        return numCentroids;
    }

};

#endif // !NUM_CENTROIDS_ALLOCATOR