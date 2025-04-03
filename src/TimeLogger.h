#ifndef TIMELOGGER_H
#define TIMELOGGER_H

#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>

class TimeLogger {
public:
    // Get the singleton instance.
    static TimeLogger& getInstance() {
        static TimeLogger instance;
        return instance;
    }

    // Record the start time of a section.
    void start(const std::string &section) {
        startTimes[section] = std::chrono::high_resolution_clock::now();
    }

    // Record the end time of a section and store the elapsed milliseconds.
    void end(const std::string &section) {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto it = startTimes.find(section);
        if (it != startTimes.end()) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - it->second).count();
            elapsedTimes[section] = duration;
        }
    }

    // Return the elapsed time for a section in seconds.
    double getTime(const std::string &section) {
        if (elapsedTimes.find(section) != elapsedTimes.end()) {
            return elapsedTimes[section];
        }
        return 0.0;
    }

    void printThroughput(const std::string &section, const size_t size_mb) {
        if (elapsedTimes.find(section) != elapsedTimes.end()) {
            const double throughput = (size_mb) / (elapsedTimes[section] / 1000.0);
            std::cout << section << ": " << throughput << " MB/s" << std::endl;
        }
    }

private:
    TimeLogger() {}  // private constructor

    // Delete copy constructor and assignment operator.
    TimeLogger(const TimeLogger&) = delete;
    TimeLogger& operator=(const TimeLogger&) = delete;

    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> startTimes;
    std::unordered_map<std::string, long long> elapsedTimes;
};

// Macros for convenient usage.
#define TIME_START(section) TimeLogger::getInstance().start(section)
#define TIME_END(section) TimeLogger::getInstance().end(section)
#define GET_TIME(section) TimeLogger::getInstance().getTime(section)
#define PRINT_THROUGHPUT(section, size) TimeLogger::getInstance().printThroughput(section, size)

#endif // TIMELOGGER_H
