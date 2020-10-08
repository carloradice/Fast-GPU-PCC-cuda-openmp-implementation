GENERAZIONE MATRICE 

1 - compilare il file random_matrix_to_file.cpp 
g++ random_matrix_to_file.cpp -o random_matrix_to_file_exec

2 - eseguire random_matrix_to_file_exec
./random_matrix_to_file_exec n m 
dove:
n = numero voxel
m = lunghezza della time serie



C-OPENMP
compilazione del file:
gcc -fopenmp CPU_side.c -o CPU_side.x -lm

esecuzione del file
./CPU_side.x N L t
N = numero di voxel
L = lunghezza time series

PYTHON-CUDA
esecuzione di CPU_side.py con python3
