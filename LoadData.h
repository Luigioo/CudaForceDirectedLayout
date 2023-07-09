#pragma once

#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>

class LoadData {
public:
    static double* readFileAndConvert(const std::string& filename, int& size);
};
