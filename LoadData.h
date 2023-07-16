#pragma once

#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>

class LoadData {
public:
    static double* readFileAndConvert(const std::string& filename, int& size);
    static int* readFileAndConvertInt(const std::string& filename, int& size, int& numNodes);
    static void writeFileWithPrecision(const std::string& filename, double* data, int size);

};
