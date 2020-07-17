#include <stdlib.h> 
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {

	std::ofstream myfile;

	myfile.open("random_matrix.txt", std::fstream::out);
	int n = atoi(argv[1]);
	int m = atoi(argv[2]);
	int max = 6;
	int min = -6;
	std::cout << "Starting writing file" << std::endl;
	for(int i=0; i<n; i++) {
		for(int j=0; j<m; j++) {
			float value = (max - min) * ( (float)rand() / (float)RAND_MAX ) + min;
			myfile << value << " ";
		}
		myfile << std::endl;
	}
	std::cout << "Ending writing file" << std::endl;
	myfile.close();

}