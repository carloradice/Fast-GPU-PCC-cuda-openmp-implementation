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

def remaining_N2(N, L, available_mem):
	x = available_mem
	temp = N
	temp *= 2
	x /= temp 
	return int(x)


def matrix_mul_cupy_3(A, B, so_far, block, N_prime, L):
	# A = np.squeeze(np.asarray(A))
	# B = np.squeeze(np.asarray(B))
	A = A[so_far*L:so_far*L+block, :]
	B = B[:, so_far*L:so_far*L+N_prime]

	print(A.shape)
	print(B.shape)

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


def cor_mat_3(BOLD, upper_tri, N, L, OOO):

	mempool = cp.get_default_memory_pool()
	pinned_mempool = cp.get_default_pinned_memory_pool()
	# calcolo memoria disponibile
	meminfo = cuda.current_context().get_memory_info()
	print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))
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

	# inizializzazione variabili
	flag = 1
	upper_size = (N-1) * N / 2
	add_uper_cpu = upper_tri
	block = OOO
	N_prime = N
	temp = 0
	temp2 = 0
	temp3 = 0
	ii=0
	pak = 0
	so_far = 0
	cormat_fullsize = 0
	count = 1
	dev_upper = 0
	final = np.array([], np.float32)

	print("Before while")
	print("block: ", block)
	print("N_prime: ", N_prime)

	print("In while")
	while flag is 1:
		print("###### ITERAZIONE ", count, " #####")
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

		#dev_corrmat = cuda.to_device(np.zeros(cormat_fullsize, np.float32))
	
		#dev_upper = cuda.to_device(np.zeros(M1, np.float32))
		
		pak += 1

		print("so_far*L: ", so_far*L)

		#############################PROVA##########################################
		# pycuda.driver.init()
		# device = pycuda.driver.Device(0)
		# ctx = device.make_context()
		
		# BOLD_so_far = BOLD[so_far*L:so_far*L+block, :]
		# BOLD_transpose_so_far = BOLD_transpose[:, so_far*L:so_far*L+N_prime]
		# print(BOLD_so_far.shape)
		# print(BOLD_transpose_so_far.shape)

		# c = np.zeros((BOLD.shape[0], BOLD_transpose.shape[1]), np.float32)
		# # BOLD_so_far_device = cp.asarray(BOLD_so_far)
		# # BOLD_transpose_so_far_device = cp.asarray(BOLD_transpose_so_far)

		# # BOLD_so_far_device = cuda.to_device(BOLD_so_far)
		# # BOLD_transpose_so_far_device = cuda.to_device(BOLD_transpose_so_far)
		# alpha = np.float32 (1.0)                             
		# beta = np.float32 (1.0) 
		# a_gpu = gpuarray.to_gpu(BOLD_so_far.copy())
		# b_gpu = gpuarray.to_gpu(BOLD_transpose_so_far.copy())
		# c_gpu = gpuarray.to_gpu(c.copy())
		
		# h = cublas.cublasCreate()  
		

		# cublas.cublasSgemm(h, 'n', 'n', block, N_prime, L, alpha, a_gpu, L, b_gpu, L, beta, c_gpu, block)
		##########################################################################

		#################################PROVA CHE FUNZIONA #########################################
		BOLD_so_far = BOLD[so_far:so_far*L+block, :]
		BOLD_transpose_so_far = BOLD_transpose[:, so_far:so_far*L+N_prime]
		print(BOLD_so_far.shape)
		print(BOLD_transpose_so_far.shape)
		
		# calcolo memoria disponibile
		meminfo = cuda.current_context().get_memory_info()
		print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))
		# copy arrays to the device
		BOLD_so_far_device = cuda.to_device(BOLD_so_far)
		BOLD_transpose_so_far_device = cuda.to_device(BOLD_transpose_so_far)

		# calcolo memoria disponibile
		
		meminfo = cuda.current_context().get_memory_info()
		print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))
		
		# allocate memory on the device for the result
		result = np.zeros((block, N_prime), np.float32)
		print(result.nbytes)
		result_device = cuda.to_device(result)
		
		meminfo = cuda.current_context().get_memory_info()
		print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))

		# configure the blocks
		threads_per_block = (TPB, TPB)
		blocks_per_grid_x = int(math.ceil(BOLD_so_far.shape[0] / threads_per_block[0]))
		blocks_per_grid_y = int(math.ceil(BOLD_transpose_so_far.shape[1] / threads_per_block[1]))
		blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

		print("Thread per block:", threads_per_block)
		print("Blocks per grid:", blocks_per_grid, "\n")


		# start the kernel for matrix multiplication
		matmul[blocks_per_grid, threads_per_block](BOLD_so_far_device, 
												   BOLD_transpose_so_far_device, 
												   result_device)

		cuda.synchronize()

		temp2 = block
		temp2 *= N_prime

		#dev_upper = cuda.to_device(np.zeros(M1, np.float32))
		upper = np.zeros(M1, np.float32)
		print(upper.shape)
		#dev_upper = cp.asarray(upper)
		dev_upper = cuda.to_device(upper)

		meminfo = cuda.current_context().get_memory_info()
		print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))

		#result = result_device.copy_to_host()
		#print(result.shape)
		result_device = cp.asarray(result_device).reshape(-1)

		threads_per_block = 1024
		blocks_per_grid = 1 + math.ceil(((temp2-1) / threads_per_block))

		print("temp2:", temp2)
		print("threads_per_block: ", threads_per_block)
		print("blocks_per_grid: ", blocks_per_grid)

		print("dev_upper shape: ", dev_upper.shape)
		print("result_device shape: ", result_device.shape)


		matrix_to_vector_2[blocks_per_grid, threads_per_block](result_device,
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

		print("add_uper_cpu: ", add_uper_cpu.shape)
		nome_file = "/home/carlo/Documents/progetto-calcolo-scientifico/python_gpu_pcc_corr_prova" + str(count) + ".txt"
		add_uper_cpu.tofile(nome_file, sep="\n")

		final = np.append(final, add_uper_cpu)
		so_far += block

		if N_prime > block:
			N_prime = N_prime - block
			block = remaining_N2(N_prime, L, available_mem)
			if N_prime < block:
				block = N_prime

		# liberare la memoria 
		#dev_upper   		 			# cuda.to_device
		#BOLD_so_far_device 				# cuda.to_device
		#BOLD_transpose_so_far_device    # cuda.to_device
		#result_device 					# cp.asarray
		del result_device
		del dev_upper
		del BOLD_so_far_device
		del BOLD_transpose_so_far_device
		
		mempool = cp.get_default_memory_pool()
		pinned_mempool = cp.get_default_pinned_memory_pool()

		cuda.defer_cleanup()

		meminfo = cuda.current_context().get_memory_info()
		print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))


		count += 1

		return final
		###########################################################################


		#dev_corrmat = matrix_mul_cupy_3(BOLD, BOLD_transpose, so_far, block, N_prime, L)

		# cuda.synchronize()

		# temp2 = block
		# temp2 *= N_prime

		# upper_size = (N-1) * N / 2

		# dev_corrmat = cuda.to_device(dev_corrmat)
		# threads_per_block = 1024
		# blocks_per_grid = int(math.ceil(1 + ((N*N - 1) / threads_per_block)))
		# matrix_to_vector_2[blocks_per_grid, threads_per_block](dev_corrmat,
		# 	dev_upper,
		# 	block,
		# 	N_prime,
		# 	upper_size,
		# 	N,
		# 	ii,
		# 	M1)
		# cuda.synchronize()

		# ii += block

		# add_uper_cpu = dev_upper.copy_to_host()

		# temp3 += M1
		# add_uper_cpu = upper_tri + temp3
		# so_far += block

		# if N_prime > block:
		# 	N_prime = N_prime - block
		# 	block = remaining_N2(N_prime, L, available_mem)
		# 	if N_prime < block:
		# 		block = N_prime