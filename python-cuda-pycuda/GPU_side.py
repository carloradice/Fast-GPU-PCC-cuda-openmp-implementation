from numba import vectorize, cuda as cd, float32
import numpy as np
import math
import time
import cupy as cp
##### PROVA ####
import skcuda.cublas as cublas
import  pycuda.gpuarray  as  gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler
#from  skcuda.cublas  import *
###########################


def remaining_N2(N, L, available_mem):
	x = available_mem
	temp = N
	temp *= 2
	x /= temp 
	return int(x)


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


def cor_mat_2(BOLD, upper_tri, N, L):
	# preprocessing fMRI data in CPU
	start_time = time.time()
	BOLD = preprocessing(BOLD, N, L)
	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time for preprocessing: ", delta, "\n")
	
	# # calcolo matrice trasposta
	# start_time = time.time()
	# BOLD_transpose = BOLD.transpose()
	# stop_time = time.time()
	# delta = stop_time - start_time	
	# print("Running time for transpose: ", delta, "\n")
	
	alpha = np.float32(1.0)
	beta = np.float32(0.0)

	# passaggio degli oggetti su device
	BOLD_device = gpuarray.to_gpu(BOLD)
	# BOLD_transpose_device = gpuarray.to_gpu(BOLD_transpose)
	
	result = np.zeros((BOLD.shape[0], BOLD.shape[0]), np.float32)
	result_device = gpuarray.to_gpu(result)

	print("BOLD_device shape:", BOLD_device.shape)
	print("result_device shape:", result_device.shape)

	start_time = time.time()
	h = cublas.cublasCreate()
	cublas.cublasSgemm(h,
					   'T',
					   'n',
					   N,
					   N,
					   L,
					   alpha,
					   BOLD_device.gpudata,
					   L,
					   BOLD_device.gpudata,
					   L,
					   beta,
					   result_device.gpudata,
					   N)

	# result1 = result_device.get()
	# print("result1 shape: ", result1.shape)
	# nome_file = "/home/carlo/Documents/progetto-calcolo-scientifico/python_gpu_pcc_corr_prova.txt"
	# result1.tofile(nome_file, sep="\n")

	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time core function: ", delta, "\n")

	start_time = time.time()

	threads_per_block = 1024
	blocks_per_grid = int(math.ceil(1 + ((N*N - 1) / threads_per_block)))

	mod = pycuda.compiler.SourceModule("""
		__global__ void ker(float * cormat, float * upper,int n1,int n)
		{
			long idx = blockDim.x*blockIdx.x+threadIdx.x;
			long i = idx%n1;
			long j = idx/n1;
			if(i<j && i<n1 && j<n)
			{
		        long tmp=i;
		        tmp*=(i+1);
		        tmp/=2;
		        long tmp_2=i;
		        tmp_2*=n;
		        tmp_2=tmp_2-tmp;
		        tmp_2+=j;
		        tmp_2-=i;
		        upper[tmp_2-1]=cormat[j*n+i];
			}
		}
		""")

	result_device = result_device.reshape(-1)
	print("result device shape:", result_device.shape)

	upper_tri_device = gpuarray.to_gpu(upper_tri)

	funct = mod.get_function("ker")
	funct(result_device, 
		  upper_tri_device, 
		  np.int32(N),
		  np.int32(N),    
		  block=(threads_per_block, 1, 1),
          grid=(blocks_per_grid, 1)
          )
	upper_tri = upper_tri_device.get()
	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time to get upper tri: ", delta, "\n")
	cublas.cublasDestroy(h)
	return upper_tri


def cor_mat_3(BOLD, N, L, OOO):
	# pycuda initialization
	# pycuda.driver.init()
	# device = pycuda.driver.Device(0)
	# ctx = device.make_context()
	# ctx.pop()

	# calcolo memoria disponibile
	meminfo = cuda.mem_get_info()
	print("free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))
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
	# add_uper_cpu = upper_tri
	block = OOO
	N_prime = N
	temp = 0
	temp2 = 0
	temp3 = 0
	ii=0
	pak = 0
	so_far = 0
	count = 1
	dev_upper = 0

	alpha = np.float32(1.0)
	beta = np.float32(0.0)

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

		M1 = int(M1)
		
		print("M1: ", M1)
		
		pak += 1

		print("so_far*L: ", so_far*L)
		#########################################################################################

		BOLD_so_far = BOLD[so_far*L:so_far*L+block, :]
		BOLD_transpose_so_far = BOLD_transpose[:, so_far*L:so_far*L+N_prime]

		print("BOLD_so_far shape: ", BOLD_so_far.shape)
		print("BOLD_transpose_so_far shape: ", BOLD_transpose_so_far.shape)

		result = np.zeros((block, N_prime), np.float32)

		# copy arrays to the device
		BOLD_so_far_device = gpuarray.to_gpu(BOLD_so_far)
		BOLD_transpose_so_far_device = gpuarray.to_gpu(BOLD_transpose_so_far)

		print("BOLD_so_far_device shape: ", BOLD_so_far_device.shape)
		print("BOLD_transpose_so_far_device shape: ", BOLD_transpose_so_far_device.shape)
		
		# calcolo memoria disponibile
		meminfo = cuda.mem_get_info()
		print("free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))
		
		# allocate memory on the device for the result
		result_device = gpuarray.to_gpu(result)

		# calcolo memoria disponibile
		meminfo = cuda.mem_get_info()
		print("free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))
		
		h = cublas.cublasCreate()

		# cublas.cublasSgemm(h,
		# 			'n',
		# 			'n',
		# 			BOLD_so_far.shape[0],
		# 			BOLD_transpose_so_far.shape[1],
		# 			BOLD_so_far.shape[1],
		# 			alpha,
		# 			BOLD_so_far_device.gpudata,
		# 			BOLD_so_far.shape[0],
		# 			BOLD_transpose_so_far_device.gpudata,
		# 			BOLD_transpose_so_far.shape[0],
		# 			beta,
		# 			result_device.gpudata,
		# 			result.shape[0]
		# 			)

		cublas.cublasSgemm(h,
						   'n',
						   'n',
						   block,
						   N_prime,
						   L,
						   alpha,
						   BOLD_so_far_device.gpudata,
						   block,
						   BOLD_transpose_so_far_device.gpudata,
						   L,
						   beta,
						   result_device.gpudata,
						   block
						   )

		result1 = result_device.get()
		# nome_file = "/home/carlo/Documents/progetto-calcolo-scientifico/python_gpu_pcc_corr_prova.txt"
		# result1.tofile(nome_file, sep="\n")
		result1 = result1.reshape(-1)
		print("result1 shape: ", result1.shape)

		del result_device

		cublas.cublasDestroy(h)

		temp2 = block
		temp2 *= N_prime

		# calcolo memoria disponibile
		meminfo = cuda.mem_get_info()
		print("free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))
		
		result_device = gpuarray.to_gpu(result1)

		threads_per_block = 1024
		blocks_per_grid = 1 + math.ceil(((temp2-1) / threads_per_block))
		grid = (blocks_per_grid, 1)

		print("temp2:", temp2)
		print("threads_per_block: ", threads_per_block)
		print("blocks_per_grid: ", blocks_per_grid)

		upper = np.zeros(M1, np.float32)
		print("upper shape:", upper.shape)
		dev_upper = gpuarray.to_gpu(upper)

		print("dev_upper shape: ", dev_upper.shape)
		print("result_device shape: ", result_device.shape)

		mod = pycuda.compiler.SourceModule("""
			__global__ void ker2(float * cormat, float * upper,int n1,int n,long long upper_size,int N,int i_so_far,long long M1)
			{
				long long idx = blockDim.x;
				idx*=blockIdx.x;
				idx+=threadIdx.x;
				long i = idx/n;
				long j = idx%n;

				if(i<j && i<n1 && j<n)// &&i<N &&j<N && idx<(n1*n))
				{
				        long long tmp=i;
				        tmp*=(i+1);
				        tmp/=2;
				        long long tmp_2=i;
				        tmp_2*=n;
				        tmp_2=tmp_2-tmp;
				        tmp_2+=j;
				        tmp_2-=i;
				        long long indexi=n1;
				        indexi*=j;
				        indexi=indexi+i;
				        upper[tmp_2-1]=cormat[indexi];
				}
			}
  			""")

		print(result_device.dtype)
		# calcolo memoria disponibile
		meminfo = cuda.mem_get_info()
		print("free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))
		
		funct = mod.get_function("ker2")
		funct(result_device, 
			  dev_upper, 
			  np.int32(block), 
			  np.int32(N_prime),
			  np.int64(upper_size),
			  np.int32(N),
			  np.int32(ii), 
			  np.int64(M1),       
			  block=(threads_per_block, 1, 1),
              grid=grid
              )

		# matrix_to_vector_2[blocks_per_grid, threads_per_block](result_device,
		# 	dev_upper,
		# 	block,
		# 	N_prime,
		# 	upper_size,
		# 	N,
		# 	ii,
		# 	M1)

		ii += block

		add_uper_cpu = dev_upper.get()

		print("add_uper_cpu: ", add_uper_cpu.shape)
		nome_file = "/home/carlo/Documents/progetto-calcolo-scientifico/python_gpu_pcc_corr_prova" + str(count) + ".txt"
		add_uper_cpu.tofile(nome_file, sep="\n", fmt="%.7f")


		flag = 0
		#########################################################################################

		# so_far += block

		# if N_prime > block:
		# 	N_prime = N_prime - block
		# 	block = remaining_N2(N_prime, L, available_mem)
		# 	if N_prime < block:
		# 		block = N_prime

		# # liberare la memoria 
		# #dev_upper   		 			# cuda.to_device
		# #BOLD_so_far_device 				# cuda.to_device
		# #BOLD_transpose_so_far_device    # cuda.to_device
		# #result_device 					# cp.asarray
		# del result_device
		# del dev_upper
		# del BOLD_so_far_device
		# del BOLD_transpose_so_far_device
		
		# mempool = cp.get_default_memory_pool()
		# pinned_mempool = cp.get_default_pinned_memory_pool()

		# cuda.defer_cleanup()

		# meminfo = cuda.current_context().get_memory_info()
		# print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))


		# count += 1

