#from numba import vectorize, cuda as cd, float32
import numpy as np
import math
import time
#import cupy as cp
import skcuda.cublas as cublas
import  pycuda.gpuarray  as  gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler


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
	
	alpha = np.float32(1.0)
	beta = np.float32(0.0)

	# passaggio su device
	start_time = time.time()
	BOLD_device = gpuarray.to_gpu(BOLD)
	result = np.zeros((BOLD.shape[0], BOLD.shape[0]), np.float32)
	result_device = gpuarray.to_gpu(result)
	# print("BOLD_device shape:", BOLD_device.shape)
	# print("result_device shape:", result_device.shape)
	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time matrices to device: ", delta, "\n")

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
	# print("result device shape:", result_device.shape)
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


def cor_mat_3(BOLD, upper_tri, N, L, OOO):
	# calcolo memoria disponibile
	meminfo = cuda.mem_get_info()
	print("free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))
	available_mem = float(meminfo[0])
	available_mem /= np.dtype(np.float32).itemsize
	available_mem -= N * L
	# print("Available memory: ", available_mem)

	# preprocessing fMRI data in CPU
	start_time = time.time()
	BOLD = preprocessing(BOLD, N, L)
	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time for preprocessing: ", delta, "\n")


	# passaggio di BOLD in device
	start_time = time.time()
	BOLD_device = gpuarray.to_gpu(BOLD)
	stop_time = time.time()
	delta = stop_time - start_time	
	print("Running time matrices to device: ", delta, "\n")

	# calcolo memoria disponibile
	# meminfo = cuda.mem_get_info()
	# print("After BOLD_device free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))

	# inizializzazione variabili
	flag = 1
	ii=0
	upper_size = (N-1) * N / 2
	block = OOO
	N_prime = N
	temp = 0
	temp2 = 0
	temp3 = 0
	pak = 0
	so_far = 0
	count = 1
	temp4 = 0

	alpha = np.float32(1.0)
	beta = np.float32(0.0)

	while flag is 1:
		print("###### ITERAZIONE ", count, " #####")
		# calcolo memoria disponibile
		# meminfo = cuda.mem_get_info()
		# print("After BOLD_device free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))

		# print("block: ", block)
		# print("N_prime: ", N_prime)
		# checking for the last chunk
		if block == N_prime:
			flag = 0

		if pak is not 0:
			del dev_upper
			del result_device

		temp = block
		temp *= (block + 1)
		temp /= 2
		# M1 is the size of upper triangle part of chunk
		M1 = N_prime
		M1 *= block
		M1 -= temp

		M1 = int(M1)
		
		# print("M1: ", M1)
		
		pak += 1

		# print("so_far*L: ", so_far*L)
		start_time = time.time()
		result = np.zeros((block, N_prime), np.float32)
		# print("result shape: ", result.shape)

		BOLD_device = BOLD_device.reshape(-1)
		
		# allocate memory on the device for the result
		result_device = gpuarray.to_gpu(result)
		# print("result_device shape: ", result_device.shape)

		# # calcolo memoria disponibile
		# meminfo = cuda.mem_get_info()
		# print("Before cublasSgemm free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))
		stop_time = time.time()
		delta = stop_time - start_time	
		print("Running time matrices to device: ", delta, "\n")

		start_time = time.time()
		h = cublas.cublasCreate()
		cublas.cublasSgemm(h,
				   'T',
				   'n',
				   block,
				   N_prime,
				   L,
				   alpha,
				   BOLD_device[so_far*L:].gpudata,
				   L,
				   BOLD_device[so_far*L:].gpudata,
				   L,
				   beta,
				   result_device.gpudata,
				   block)
		stop_time = time.time()
		delta = stop_time - start_time	
		print("Running time core function: ", delta, "\n")

		temp2 = block
		temp2 *= N_prime

		# calcolo memoria disponibile
		# meminfo = cuda.mem_get_info()
		# print("free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))
		
		# result_device = gpuarray.to_gpu(result1)

		start_time = time.time()
		threads_per_block = 1024
		blocks_per_grid = 1 + math.ceil(((temp2-1) / threads_per_block))
		grid = (blocks_per_grid, 1)

		# print("temp2:", temp2)
		# print("threads_per_block: ", threads_per_block)
		# print("blocks_per_grid: ", blocks_per_grid)

		upper = np.zeros(M1, np.float32)
		# print("upper shape:", upper.shape)
		dev_upper = gpuarray.to_gpu(upper)

		# print("dev_upper shape: ", dev_upper.shape)
		# print("result_device shape: ", result_device.shape)

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
		# calcolo memoria disponibile
		# meminfo = cuda.mem_get_info()
		# print("free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))
		
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

		temp3+=M1
		# print("upper_tri shape:", upper_tri.shape)
		upper_tri[temp4:temp3] = dev_upper.get()
		stop_time = time.time()
		delta = stop_time - start_time	
		print("Running time to get upper tri: ", delta, "\n")

		temp4 += M1
		ii += block

		cublas.cublasDestroy(h)

		so_far += block

		if N_prime > block:
			N_prime = N_prime - block
			block = remaining_N2(N_prime, L, available_mem)
			if N_prime < block:
				block = N_prime

		count += 1

	# liberare la memoria 
	del BOLD_device
	del result_device
	del dev_upper

	# calcolo memoria disponibile
	# meminfo = cuda.mem_get_info()
	# print("free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))

	return upper_tri