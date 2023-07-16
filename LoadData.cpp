#include "LoadData.h"

#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>

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

int* LoadData::readFileAndConvertInt(const std::string& filename, int& size, int& numNodes) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "File not found!" << std::endl;
        return nullptr;
    }

    std::string line;
    // Read the first line containing the array data
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<int> data;
        int num;
        while (ss >> num) {
            data.push_back(num);
        }

        size = data.size();
        int* arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = data[i];
        }

        // Read the second line containing the number of nodes
        if (std::getline(file, line)) {
            std::stringstream ss2(line);
            if (ss2 >> numNodes) {
                return arr;
            }
        }

        // Error occurred while reading the number of nodes
        delete[] arr;
        std::cout << "Failed to read the number of nodes." << std::endl;
        return nullptr;
    }

    // Error occurred while reading the array data
    std::cout << "Failed to read the array data." << std::endl;
    return nullptr;
}

void LoadData::writeFileWithPrecision(const std::string& filename, double* data, int size) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << "Unable to open file!" << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < size; i++) {
        file << data[i] << '\n';
    }
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
