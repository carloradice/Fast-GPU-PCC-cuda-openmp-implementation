import numpy as np
from numba import cuda
from GPU_side import cor_mat_2, cor_mat_3
import time

gpu = cuda.gpus.lst[0]

# questa funzione controlla che ci sia abbastanza memoria per calcolare la matrice
def remaining_mem(N, L, flag):
	meminfo = cuda.current_context().get_memory_info()
	print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))
	available_mem = float(meminfo[0])
	available_mem /= np.dtype(np.float32).itemsize
	NL = N * L
	if flag is 0:
		available_mem -= NL
	x = available_mem
	temp = N * 2
	x /= temp
	return int(x)


def remaining_N2(N, L, available_mem):
	x = available_mem
	temp = N
	temp *= 2
	x /= temp 
	return int(x)



def main():
	# salvo la matrice come numpy array
	BOLD = np.array(np.loadtxt("/home/carlo/Documents/progetto-calcolo-scientifico/random_matrix.txt"), np.float32)

	# ottengo il numero di voxel e la lunghezza della time serie dalle dimensioni della matrice
	N = BOLD.shape[0]
	L = BOLD.shape[1]
	print("Number of voxels: ", N)
	print("Length of time series: ", L)

	# creazione matrice triangolare superiore e inizializzazione a zero
	size = int(((N-1) * N) / 2)
	upper_tri = np.zeros(size, np.float32)
	
	# calcolo memoria necessaria per la matrice
	rem_mem = remaining_mem(N, L, 0)
	print("Remaining mem:", rem_mem)

	print("Computing correlations ...")
	# se la matrice occupa meno memoria del totale
	if N <= rem_mem:
		print("cor_mat_2")
		start_time = time.time()

		upper_tri = cor_mat_2(BOLD, upper_tri, N, L)
		
		stop_time = time.time()
		delta = stop_time - start_time	
		print("Running time for computing correlations: ", delta, "\n")
	# se la matrice occupa più memoria del totale
	if N > rem_mem:
		print("cor_mat_3")
		start_time = time.time()

		upper_tri = cor_mat_3(BOLD, upper_tri, N, L, rem_mem)

		stop_time = time.time()
		delta = stop_time - start_time	
		print("Running time for computing correlations: ", delta, "\n")


	#np.savetxt("/home/carlo/Documents/progetto-calcolo-scientifico/python_gpu_pcc_corr.txt", upper_tri, delimiter=" ", fmt="%.7f")


if __name__ == '__main__':
    main()