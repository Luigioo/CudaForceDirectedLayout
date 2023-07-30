#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>


using namespace std;

class Config {

public:
	string input_path;
	string output_path;
	int processor;
	int iteration;

	Config() {
		input_path = "./graph_data/grid_nodes_100_input.txt";
		output_path = "./position_data/outputCuda.txt";
		processor = 0;
		iteration = 500;
	}
	~Config() {

	}
    // This function parses the command line arguments
    void ParseArgs(int argc, char** argv) {
        // If no arguments are provided, use default values
        if (argc == 1) {
            cout << "No arguments provided. The program will continue with default values." << endl;
            return;
        }

        // Loop over the arguments (starting from 1 because argv[0] is the program name)
        for (int i = 1; i < argc; i++) {
            // If the argument is "-i", it is followed by the input path
            if (string(argv[i]) == "-i") {
                input_path = string(argv[i + 1]);
                i++;
                cout << "Input path set to: " << input_path << endl;
            }
            // If the argument is "-o", it is followed by the output path
            else if (string(argv[i]) == "-o") {
                output_path = string(argv[i + 1]);
                i++;
                cout << "Output path set to: " << output_path << endl;
            }
            // If the argument is "-p", it is followed by the processor number
            // 0 stands for GPU, 1 stands for CPU
            else if (string(argv[i]) == "-p") {
                processor = atoi(argv[i + 1]);
                i++;
                cout << "Processor number set to: " << processor << endl;
            }
            // If the argument is "-t", it is followed by the iteration number
            else if (string(argv[i]) == "-t") {
                iteration = atoi(argv[i + 1]);
                i++;
                cout << "Iteration number set to: " << iteration << endl;
            }
            // If the argument does not match any expected flags, it is invalid
            else {
                cout << "Invalid arguments. Expected flags are -i (input path), -o (output path), -p (processor, 0 for GPU, 1 for CPU), and -t (iteration number)." << endl;
                exit(1);
            }
        }
    }


};