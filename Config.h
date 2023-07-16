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
		input_path = "./graph_data/large_grid.txt";
		output_path = "./position_data/outputCuda.txt";
		processor = 0;
		iteration = 50;
	}
	~Config() {

	}
	void ParseArgs(int argc, char** argv) {
		if(argc==1){
			cout << "No arguments, continue with default values" << endl;
			return;
		}
		for (int i = 1; i < argc; i++) {
			if (string(argv[i]) == "-i") {
				input_path = string(argv[i + 1]);
				i++;
			}
			else if (string(argv[i]) == "-o") {
				output_path = string(argv[i + 1]);
				i++;
			}
			else if (string(argv[i]) == "-p") {
				processor = atoi(argv[i + 1]);
				i++;
			}
			else if (string(argv[i]) == "-t") {
				iteration = atoi(argv[i + 1]);
				i++;
			}
			else {
				cout << "Invalid arguments" << endl;
				exit(1);
			}
		}
	}

};