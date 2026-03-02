#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <chrono>

struct Metric {
    std::string metricName;
    std::string metricValue;
};
class MetricLogger {
    std::vector<Metric> metrics;
    std::string logFilename;
public:
    MetricLogger(const std::string& logFilename): logFilename(logFilename) {
        std::ifstream file(logFilename);  // 打开文件
        if (!file.is_open()) {
            std::cerr << "File Open Failed : " << logFilename << std::endl;
            assert(false);
        }

        std::string line;
        std::getline(file, line);
        std::stringstream ss(line);  // 将行内容存入字符串流
        std::string token;
        while (std::getline(ss, token, ',')) {  // 按逗号分隔
            Metric metric;
            metric.metricName = token;
            metric.metricValue = "N.A.";
            metrics.push_back(metric);
        }

        file.close();  // 关闭文件
    }

    template<typename DType>
    void addLog(const std::string& metricName, const DType& value) {
        std::ostringstream oss;
        oss << value;

        for(Metric& metric : metrics) {
            if(metric.metricName == metricName) {
                metric.metricValue = oss.str();
                return;
            }
        }
        assert(false && "Metric Name NOT Existed!");
    }

    template<typename T>
    void addLogVector(const std::string& metricName, const std::vector<T>& value) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < value.size(); ++i) {
            if (i > 0) oss << "; ";
            oss << value[i];
        }
        oss << "]";

        for (Metric& metric : metrics) {
            if (metric.metricName == metricName) {
                metric.metricValue = oss.str();
                return;
            }
        }
        assert(false && "Metric Name NOT Existed!");
    }

    void writeLog() {
        int fd = open(logFilename.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0666);
        if (fd == -1) {
            perror("无法打开文件");
            return;
        }

        struct flock lock;
        lock.l_type = F_WRLCK;  // 写锁
        lock.l_whence = SEEK_SET;
        lock.l_start = 0;
        lock.l_len = 0;  // 锁定整个文件

        // 循环等待直到成功加锁
        while (fcntl(fd, F_SETLK, &lock) == -1) {
            std::cerr << "Metric File is locked, waiting..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));  // 等待 500 毫秒后重试
        }

        // 使用 FILE* 写入文件
        FILE* file = fdopen(fd, "a");
        if (file == nullptr) {
            assert(false && "Metric File open failed");
        }

        // 将字符串按逗号拼接并写入文件
        for (size_t i = 0; i < metrics.size(); ++i) {
            fprintf(file, "%s", metrics[i].metricValue.c_str());
            if (i != metrics.size() - 1) {  // 如果不是最后一个元素，添加逗号
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");  // 添加换行符

        // 解锁文件
        lock.l_type = F_UNLCK;
        if (fcntl(fd, F_SETLK, &lock) == -1) {
            assert(false && "Metric File unlock failed!");
        }

        fclose(file);  // 关闭文件
    }
};