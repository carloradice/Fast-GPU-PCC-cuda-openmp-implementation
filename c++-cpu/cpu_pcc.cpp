#include <iostream>
#include <fstream>
#include <vector>

int main(int arg, char *argv[]) {
	
	// rows
	int N = atoi(argv[1]);
	// columns
	int L = atoi(argv[2]);
	// write type
	char wr = *(argv[3]);
	std::cout << "Number of voxels: " << N 
			  << "  " << "Length of time series: "
			  << L << "\n\n";

	// open file to read matrix
	std::ifstream myReadFile;
    myReadFile.open("/home/carlo/Documents/progetto-calcolo-scientifico/random_matrix.txt");
    std::vector<std::vector<float>> matrix2d (N, std::vector<float>(L, 0));
    //float *matrix = new float [L * N];
    for(auto k = 0; k < N; k++) {
       	for(auto l = 0; l < L; l++) {
       		// v[k * L + l] = matrix[k][l]
            myReadFile >> matrix2d[k][l];
            std::cout << matrix2d[k][l] << " ";
        }
        std::cout << "\n";
    }
    myReadFile.close();
    
    std::cout << "\n";
    
    std::vector<std::vector<float>> transpose2d (L, std::vector<float>(N, 0));
    for(auto l = 0; l < L; l++) {
       	for(auto k = 0; k < N; k++) {
            transpose2d[l][k] = matrix2d[k][l];
            std::cout << transpose2d[l][k] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\n";
    
}