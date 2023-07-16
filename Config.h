#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>


using namespace std;

class Config {

public:
	Config() {
		input_path = "./graph_data.txt";
		output_path = "./position_data/output.txt";
		processor = 0;
		iteration = 50;
	}
	~Config() {

	}
	void ParseArgs(int argc, char** argv) {
		int opt;
		const char* str = "p:l:m:o:s:t:c:a:";
		while ((opt = getopt(argc, argv, str)) != -1) {
			switch (opt) {
			case 'p':
				processor = atoi(optarg);
				break;
			case 'i':
				input_path = string(optarg);
				break;
			case 'o':
				output_path = string(optarg);
				break;
			case 't':
				iteration = atoi(optarg);
				break;
			default:
				cout << "Invalid arguments" << endl;
				exit(1);
			}
		}
		
	}

private:
	string input_path;
	string output_path;
	int processor;
	int iteration;


};