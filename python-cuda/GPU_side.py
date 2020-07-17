from numba import vectorize, cuda, float32
import numpy as np
import math
import time
import cupy as cp
##### PROVA ####
import skcuda.cublas as cublas
import  pycuda.gpuarray  as  gpuarray
import pycuda.driver
###########################

# threads per block
TPB = 16

gpu = cuda.gpus.lst[0]

def matrix_mul_cupy_3(A, B, so_far, block, N_prime, L):
	# A = np.squeeze(np.asarray(A))
	# B = np.squeeze(np.asarray(B))
	A = A[so_far*L:so_far*L+block, :]
	B = B[:, so_far*L:so_far*L+N_prime]

	print(A.shape)
	print(B.shape)
	# A = A[so_far*L:]
	# A = A[:block*L]
	# B = B[so_far*L:]
	# B = B[:block*N_prime]


	A_device = cp.asarray(A)
	B_device = cp.asarray(B)
	C = np.zeros((A.shape[0], B.shape[1]), np.float32)
	print(C.shape)
	C_device = cp.asarray(C)

	#C_device =
	C = cp.matmul(A_device, B_device)
	#C = cp.asnumpy(C_device)

	return C


def matrix_mul_cupy(A, B):
	A_device = cp.asarray(A)
	B_device = cp.asarray(B)
	C = np.array((A.shape[0], B.shape[1]))
	C_device = cp.asarray(C)

	C_device = cp.matmul(A_device, B_device)
	C = cp.asnumpy(C_device)

	return C


# preprocessing della matrice in CPU
def preprocessing(BOLD, N, L):
	for i in range(0, N):
		B = BOLD[i,:]
		sum1 = sum(B) / L
		sum2 = np.sqrt(sum(np.power((B - sum1), 2)))
		if sum2 is not 0:
			B = (B - sum1) / sum2
		else:
			B = 0
		BOLD[i, :] = B
	return BOLD


@cuda.jit
def matrix_to_vector(result_device, upper_tri_device, N):
	idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x 
	i = int(idx % N)
	j = int(idx / N)
	if i < j and i < N and j < N:
		tmp = i
		tmp *= (i+1)
		tmp /= 2
		tmp_2 = i
		tmp_2 *= N
		tmp_2 = tmp_2 - tmp
		tmp_2 += j
		tmp_2 -= i
		upper_tri_device[int(tmp_2)-1] = result_device[i][j]


@cuda.jit
def matrix_to_vector_2(cormat, upper, n1, n, upper_size, N, i_so_far, M1):
	idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x 
	i = int(idx / n)
	j = int(idx % n)
	if i < j and i < n1 and j < n:
		tmp = i
		tmp *= (i+1)
		tmp /= 2
		tmp_2 = i
		tmp_2 *= n
		tmp_2 = tmp_2 - tmp
		tmp_2 += j
		tmp_2 -= i
		indexi = n1
		indexi *= j
		indexi = indexi + i
		upper[int(tmp_2)-1] = cormat[indexi]


@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
    	tmp = 0.0
    	for k in range(A.shape[1]):
    		tmp += A[i, k] * B[k, j]
    		C[i, j] = tmp


@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


def cor_mat_2(BOLD, upper_tri, N, L):
	# preprocessing fMRI data in CPU
	start_time = time.time()
	BOLD = preprocessing(BOLD, N, L)
	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time for preprocessing: ", delta, "\n")
	
	# calcolo matrice trasposta
	start_time = time.time()
	BOLD_transpose = BOLD.transpose()
	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time for transpose: ", delta, "\n")
	
	start_time = time.time()
	# copy arrays to the device
	#BOLD_device = cuda.to_device(BOLD)
	#BOLD_device_transpose = cuda.to_device(BOLD_transpose)

	# allocate memory on the device for the result
	#result_device = cuda.device_array((N, N))

	# configure the blocks
	threads_per_block = (TPB, TPB)
	blocks_per_grid_x = int(math.ceil(BOLD.shape[0] / threads_per_block[0]))
	blocks_per_grid_y = int(math.ceil(BOLD_transpose.shape[1] / threads_per_block[1]))
	blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

	print("Thread per block:", threads_per_block)
	print("Blocks per grid:", blocks_per_grid, "\n")


	# start the kernel for matrix multiplication
	# matmul[blocks_per_grid, threads_per_block](BOLD_device, 
	# 										   BOLD_device_transpose, 
	# 										   result_device)

	# start the kernel for fast matrix multiplication
	# fast_matmul[blocks_per_grid, threads_per_block](BOLD_device, 
	# 										   		BOLD_device_transpose, 
	# 										   		result_device)

	result_device = matrix_mul_cupy(BOLD, BOLD_transpose)

	#cuda.synchronize()
	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time core function: ", delta, "\n")

	start_time = time.time()
	
	# funziona meglio con matrici piccole (CPU)
	# upper_tri = result_device.copy_to_host()
	# m = upper_tri.shape[0]
	# r,c = np.triu_indices(m,1)
	# upper_tri = upper_tri[r, c]
	
	# funziona meglio con matrici grandi (parallelo)
	result_device = cuda.to_device(result_device)
	upper_tri_device = cuda.to_device(upper_tri)
	threads_per_block = 1024
	blocks_per_grid = int(math.ceil(1 + ((N*N - 1) / threads_per_block)))
	matrix_to_vector[blocks_per_grid, threads_per_block](result_device,
	 													 upper_tri_device,
	 													 N)
	upper_tri = upper_tri_device.copy_to_host()

	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time to get upper tri: ", delta, "\n")

	return upper_tri

def cor_mat_3(BOLD, upper_tri, N, L, OOO):
	# calcolo memoria disponibile
	meminfo = cuda.current_context().get_memory_info()
	available_mem = float(meminfo[0])
	available_mem /= np.dtype(np.float32).itemsize
	available_mem -= N * L

	print("Available memory: ", available_mem)

	# preprocessing fMRI data in CPU
	start_time = time.time()
	BOLD = preprocessing(BOLD, N, L)
	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time for preprocessing: ", delta, "\n")

	# calcolo matrice trasposta
	start_time = time.time()
	BOLD_transpose = BOLD.transpose()
	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time for transpose: ", delta, "\n")

	temp = 0
	temp2 = 0
	temp3 = 0
	ii=0
	flag = 1
	pak = 0
	so_far = 0
	add_uper_cpu = upper_tri
	block = OOO
	N_prime = N
	cormat_fullsize = 0

	print("Before while")
	print("block: ", block)
	print("N_prime: ", N_prime)

	print("In while")
	while flag is 1:
		# checking for the last chunk
		if block == N_prime:
			flag = 0

		temp = block
		temp *= (block + 1)
		temp /= 2
		# M1 is the size of upper triangle part of chunk
		M1 = N_prime
		M1 *= block
		M1 -= temp

		cormat_fullsize = block
		cormat_fullsize *= N_prime
		M1 = int(M1)

		print("cormat_fullsize: ", cormat_fullsize)
		
		print("M1: ", M1)

		dev_corrmat = cuda.to_device(np.zeros(cormat_fullsize, np.float32))
	
		dev_upper = cuda.to_device(np.zeros(M1, np.float32))
		
		pak += 1

		print("so_far*L: ", so_far*L)

		#############################PROVA##########################################
		pycuda.driver.init()
		device = pycuda.driver.Device(0)
		ctx = device.make_context()
		
		BOLD_so_far = BOLD[so_far*L:so_far*L+block, :]
		BOLD_transpose_so_far = BOLD_transpose[:, so_far*L:so_far*L+N_prime]
		print(BOLD_so_far.shape)
		print(BOLD_transpose_so_far.shape)

		c = np.zeros((BOLD.shape[0], BOLD_transpose.shape[1]), np.float32)
		# BOLD_so_far_device = cp.asarray(BOLD_so_far)
		# BOLD_transpose_so_far_device = cp.asarray(BOLD_transpose_so_far)

		# BOLD_so_far_device = cuda.to_device(BOLD_so_far)
		# BOLD_transpose_so_far_device = cuda.to_device(BOLD_transpose_so_far)
		alpha = np.float32 (1.0)                             
		beta = np.float32 (1.0) 
		a_gpu = gpuarray.to_gpu(BOLD_so_far.copy())
		b_gpu = gpuarray.to_gpu(BOLD_transpose_so_far.copy())
		c_gpu = gpuarray.to_gpu(c.copy())
		
		h = cublas.cublasCreate()  
		

		cublas.cublasSgemm(h, 'n', 'n', block, N_prime, L, alpha, a_gpu, L, b_gpu, L, beta, c_gpu, block)
		##########################################################################
		#dev_corrmat = matrix_mul_cupy_3(BOLD, BOLD_transpose, so_far, block, N_prime, L)

		cuda.synchronize()

		temp2 = block
		temp2 *= N_prime

		upper_size = (N-1) * N / 2

		dev_corrmat = cuda.to_device(dev_corrmat)
		threads_per_block = 1024
		blocks_per_grid = int(math.ceil(1 + ((N*N - 1) / threads_per_block)))
		matrix_to_vector_2[blocks_per_grid, threads_per_block](dev_corrmat,
			dev_upper,
			block,
			N_prime,
			upper_size,
			N,
			ii,
			M1)
		cuda.synchronize()

		ii += block

		add_uper_cpu = dev_upper.copy_to_host()

		temp3 += M1
		add_uper_cpu = upper_tri + temp3
		so_far += block

		if N_prime > block:
			N_prime = N_prime - block
			block = remaining_N2(N_prime, L, available_mem)
			if N_prime < block:
				block = N_prime