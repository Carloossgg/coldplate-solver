#pragma once

#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <mutex>
#include <iomanip>

class GlobalTimer {
public:
    static GlobalTimer& instance() {
        static GlobalTimer inst;
        return inst;
    }

    void addTime(const std::string& name, double seconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        times_[name] += seconds;
        counts_[name]++;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        times_.clear();
        counts_.clear();
    }

    void printReport() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "\n=========================================" << std::endl;
        std::cout << "          PERFORMANCE PROFILE            " << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << std::left << std::setw(30) << "Section" 
                  << std::right << std::setw(12) << "Total (ms)" 
                  << std::setw(10) << "Calls" 
                  << std::setw(12) << "Avg (ms)" << std::endl;
        std::cout << "----------------------------------------------------------------" << std::endl;

        // Sort by total time descending
        std::vector<std::pair<std::string, double>> sortedTimes(times_.begin(), times_.end());
        std::sort(sortedTimes.begin(), sortedTimes.end(), 
            [](const auto& a, const auto& b) { return a.second > b.second; });

        for (const auto& kv : sortedTimes) {
            const std::string& name = kv.first;
            double totalSec = kv.second;
            long long count = counts_[name];
            double avgMs = (totalSec * 1000.0) / count;
            
            std::cout << std::left << std::setw(30) << name 
                      << std::right << std::setw(12) << std::fixed << std::setprecision(2) << (totalSec * 1000.0)
                      << std::setw(10) << count 
                      << std::setw(12) << avgMs << std::endl;
        }
        std::cout << "=========================================\n" << std::endl;
    }

private:
    std::map<std::string, double> times_;
    std::map<std::string, long long> counts_;
    std::mutex mutex_;
};

class ScopedTimer {
public:
    ScopedTimer(const std::string& name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start_;
        GlobalTimer::instance().addTime(name_, diff.count());
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
