import numpy as np
import time

matrix = np.loadtxt("/home/carlo/Documents/progetto-calcolo-parallelo/random_matrix.txt")

# library function 
start_time = time.time()
corrmatrix = np.corrcoef(matrix)

m = corrmatrix.shape[0]
r,c = np.triu_indices(m,1)
triu_corrmatrix = corrmatrix[r, c]

stop_time = time.time()

delta = stop_time - start_time

print("Library function")
print("Running time for computing correlations: ", delta, "\n")

# print(triu_corrmatrix)

np.savetxt("/home/carlo/Documents/progetto-calcolo-parallelo/python_cpu_pcc_corr.txt", triu_corrmatrix, delimiter=" ", fmt="%.7f")

# naive function
start_time = time.time()

N = matrix.shape[0]
L = matrix.shape[1]

for i in range(0, N):
	sum1 = 0
	sum2 = 0
	for l in range(0, L):
		sum1 += matrix[i][l]
	sum1 /= L
	for l in range(0, L):
		sum2 += (matrix[i][l] - sum1) * (matrix[i][l] - sum1)
	sum2 = np.sqrt(sum2)
	for l in range(0, L):
		if sum2 is not 0:
			matrix[i][l]= (matrix[i][l] - sum1) / sum2
		else:
			matrix[i][l] = 0

transpose = matrix.transpose()

corrmatrix = np.matmul(matrix, transpose)

m = corrmatrix.shape[0]
r,c = np.triu_indices(m,1)
triu_corrmatrix = corrmatrix[r, c]

stop_time = time.time()

delta = stop_time - start_time

print("Naive function")
print("Running time for computing correlations: ", delta)

#print(triu_corrmatrix)
#print(triu_corrmatrix.size)