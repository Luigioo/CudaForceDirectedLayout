#include "LoadData.h"

#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>

#include "LoadData.h"

double* LoadData::readFileAndConvert(const std::string& filename, int& size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "File not found!" << std::endl;
        return nullptr;
    }

    std::vector<double> data;
    double num;
    while (file >> num) {
        data.push_back(num);
    }

    size = data.size();
    double* arr = new double[size];
    for (int i = 0; i < size; i++) {
        arr[i] = data[i];
    }

    return arr;
}

//class LoadData {
//
//public:
//    static double* readFileAndConvert(const std::string& filename, int& size) {
//        std::ifstream file(filename);
//        if (!file.is_open()) {
//            std::cout << "File not found!" << std::endl;
//            return nullptr;
//        }
//
//        std::vector<double> data;
//        double num;
//        while (file >> num) {
//            data.push_back(num);
//        }
//
//        size = data.size();
//        double* arr = new double[size];
//        for (int i = 0; i < size; i++) {
//            arr[i] = data[i];
//        }
//
//        return arr;
//    }
//
//};


//int main() {
//    int size = 0;
//    double* arr = readFileAndConvert("array_data.txt", size);
//
//    for (int i = 0; i < size; ++i) {
//        std::cout << arr[i] << " ";
//    }
//    delete[] arr; // Don't forget to delete!
//
//
//
//
//    return 0;
//}
