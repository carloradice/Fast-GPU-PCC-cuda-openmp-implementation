compilazione del file:
gcc -fopenmp CPU_side.c -o CPU_side.x -lm

esecuzione del file
./CPU_side.x N L t
N = numero di voxel
L = lunghezza time series
