#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


long long remaining_mem(int N, int L) {
	long long available_mem = 10000000000;
	// division by float size = 4 byte
	available_mem /= 4;
	available_mem -= N * L;
	float temp = N * 2;
	available_mem /= temp;
	printf("available_mem: %lld \n", available_mem);
	return available_mem;
}


float remaining_N2(int N_prime, int L, long long rem_mem) {
	
	long long x = rem_mem;
    long long temp = N_prime;
    temp *= 2;
    x /= temp;
    printf("rem mem: %lld \n", rem_mem);
    printf("N_prime: %d \n", N_prime);
    printf("x: %lld \n", x);
    return x;
}


void preprocessing(float * BOLD, int N, int L) {
	for(size_t i = 0; i < N; i++) {
		float * row = BOLD + i * L;
		float sum1 = 0, sum2 = 0;
		for(size_t l = 0; l < L; l++) {
			sum1 += row[l];
		}
		sum1 /= L;
		for(size_t l = 0; l < L; l++) {
			sum2 += (row[l] - sum1) * (row[l] - sum1);
		}
		sum2 = sqrt(sum2);
		for(size_t l = 0; l < L; l++) {
			if(sum2 != 0) {
				row[l] = (row[l] - sum1) / sum2;
			}
			else {
				if(sum2 == 0) {
					row[l] = 0;
				}
			}
		}
	}
}


void CorMat_2(float * BOLD, float * upper_tri, int N, int L) {

	size_t i, j, k;
	float * BOLD_transpose;
	float * result;

	double start_time, stop_time;

	// preprocessing fMRI data
	start_time = omp_get_wtime(); 
	preprocessing(BOLD, N, L);
	stop_time = omp_get_wtime();
	printf("Running time for preprocessing: %f \n", stop_time - start_time);

	BOLD_transpose = (float *) malloc((L * N) * sizeof(float));
	result = (float *) malloc((N * N) * sizeof(float));

	// get BOLD_transpose
	start_time = omp_get_wtime(); 
	float temp;
	for(i = 0; i < N; i++) {
		for(j = 0; j < L; j++) {
			temp = BOLD[i * L + j];
			BOLD_transpose[j * N + i] = temp;
		}
	}
	stop_time = omp_get_wtime(); 
	printf("Running time for transpose: %f \n", stop_time - start_time);

	// matrix product
	start_time = omp_get_wtime(); 
 	# pragma omp parallel shared (N, L, BOLD, BOLD_transpose, result) private (i, j, k)
	{
		# pragma omp for
		for(i = 0; i < N; i++) {
			for(j = 0; j < N; j++) {
				result[i * N + j] = 0;
				for(k = 0; k < L; k++) {
					result[i * N + j] += BOLD[i * L + k] * BOLD_transpose[k * N + j];
				}	
			}
		}
	}
	stop_time = omp_get_wtime();
	printf("Running time core function: %f \n", stop_time - start_time);

	// get upper triangular matrix
	start_time = omp_get_wtime(); 
	# pragma omp parallel shared (N, upper_tri, result) private (i, j)
	{
		int idx = 0;
		for(i = 0; i < N; i++) {
			for(j = 0; j < N; j++) {
				idx = (N * i) + j - ((i * (i+1)) / 2);
				idx -= (i+1);
				if(i < j) {
					upper_tri[idx] = result[i * N + j];
				}
			}
		}
	}
	stop_time = omp_get_wtime();
	printf("Running time to get upper tri: %f \n", stop_time - start_time);

	free(result);
	free(BOLD_transpose);
}


void CorMat_3(float * BOLD, float* upper_tri, int N, int L, long long OOO) {
	
	size_t i, j, k;

	float * BOLD_transpose;
	float * result;
	float * BOLD_section;
	float * upper_section;

	long long available_mem = 10000000000;
	available_mem /= sizeof(float);
	available_mem -= (N*L);

	printf("Available memory: %lld \n", available_mem);

	double start_time, stop_time;

	// preprocessing fMRI data
	start_time = omp_get_wtime(); 
	preprocessing(BOLD, N, L);
	stop_time = omp_get_wtime();
	printf("Running time for preprocessing: %f \n", stop_time - start_time);

	BOLD_transpose = (float *) malloc((L * N) * sizeof(float));

	// get BOLD_transpose
	start_time = omp_get_wtime(); 
	float temporary;
	for(i = 0; i < N; i++) {
		for(j = 0; j < L; j++) {
			temporary = BOLD[i * L + j];
			BOLD_transpose[j * N + i] = temporary;
		}
	}
	stop_time = omp_get_wtime(); 
	printf("Running time for transpose: %f \n", stop_time - start_time);

	int flag = 1;
	int count = 0;
	int block = OOO;
	int N_prime = N;
	int so_far = 0;
	int pak = 0;
	long long M1, temp;
	int limit = 0;
	long long cormat_fullsize;

	while(flag == 1) {
		printf("########## ITERAZIONE %d \n", count);
		
		if(block == N_prime) {
			flag = 0;
		}

		temp = block;
		temp *= (block + 1);
		temp /= 2;
		M1 = N_prime;
		M1 *= block;
		M1 -= temp;

		if(pak != 0) {
			free(result); //result = devCormat in C++ CUDA
			free(upper_section);
			free(BOLD_transpose);
		}

		cormat_fullsize = block;
		cormat_fullsize *= N_prime;

		printf("block: %d \n", block);
		printf("N_prime: %d \n", N_prime);
		printf("cormat_fullsize: %lld \n", cormat_fullsize);

		result = (float *) malloc(cormat_fullsize * sizeof(float));
		upper_section = (float *) malloc(M1 * sizeof(float));
		
		BOLD_section = (float *) malloc(N_prime * L * sizeof(float));

		
		if (count != 0) {
			printf("count > 0: get BOLD_section\n");
			// get BOLD_section transpose
			float temporary;
			for(i = 0; i < N_prime; i++) {
				for(j = 0; j < L; j++) {
					temporary = BOLD[(so_far * L) + (i * L) + j];
					BOLD_section[j * N_prime + i] = temporary;
				}
			}
		}

		pak++;

		// matrix product
		start_time = omp_get_wtime(); 
		if (count != 0) {
			// caso passo: da dopo la prima iterazione
	 		# pragma omp parallel shared (block, N_prime, L, so_far, BOLD, BOLD_section, result) private (i, j, k)
			{
				# pragma omp for
				for(i = 0; i < block; i++) {
					for(j = 0; j < N_prime; j++) {
						result[i * N_prime + j] = 0;
						for(k = 0; k < L; k++) {
							result[i * N_prime + j] += BOLD[(so_far * L) + (i * L) + k] * BOLD_section[(k * N_prime) + j];
						}	
					}
				}
				
			}
		}
		else {
			// caso base: prima iterazione
			# pragma omp parallel shared (block, N_prime, L, so_far, BOLD, BOLD_transpose, result) private (i, j, k)
			{
				# pragma omp for
				for(i = 0; i < block; i++) {
					for(j = 0; j < N_prime; j++) {
						result[i * N_prime + j] = 0;
						for(k = 0; k < L; k++) {
							result[i * N_prime + j] += BOLD[L * i + k] * BOLD_transpose[k * N_prime + j];
						}	
					}
				}
			}
		}
		stop_time = omp_get_wtime();
		printf("Running time core function: %f \n", stop_time - start_time);

		// get upper triangular matrix
		start_time = omp_get_wtime(); 
		# pragma omp parallel shared (block, N_prime, upper_section, result) private (i, j)
		{
			int idx = 0;
			for(i = 0; i < block; i++) {
				for(j = 0; j < N_prime; j++) {
					idx = (N_prime * i) + j - ((i * (i+1)) / 2);
					idx -= (i+1);
					if(i < j) {
						upper_section[idx] = result[i * N_prime + j];
					}
				}
			}
		}
		stop_time = omp_get_wtime();
		printf("Running time to get upper tri: %f \n", stop_time - start_time);

		long long M11 = (N-1);
		M11 *= N;
		M11 /= 2;
		long long idx = 0;
		for(long long index = 0; index < M11; index++) {
			if(upper_tri[index] == 0.000000 && idx < M1) {
				upper_tri[index] = upper_section[idx];
				idx++;
			}
		}
		
		so_far += block;

		if(N_prime>block) {
            N_prime = N_prime-block;
            block=remaining_N2(N_prime, L, available_mem);
          
            if(N_prime < block) {//checking last chunk
             	block = N_prime;
             	printf("END \n");
         	}

	    }

	    printf("block: %d \n", block);
	    printf("N_prime: %d \n", N_prime);

		count++;
	}

	free(BOLD_section);
	free(result);
	free(upper_section);

}


int main(int argc, char **argv) {

	int N, L;
	char wr;

	N = (int) strtol(argv[1], (char **)NULL, 10);
	L = (int) strtol(argv[2], (char **)NULL, 10);
	wr = *argv[3];

	printf("Number of voxels: %d, Length of time series: %d \n", N, L);

	double start_time, stop_time;

	float * BOLD;
	BOLD = (float *) malloc((N* L) * sizeof(float));

	printf("Allocated matrix dimension: %ld \n", N*L * sizeof(float));
	
	// open file to read matrix
	FILE *fp;
	fp = fopen("/home/carlo/Documents/progetto-calcolo-parallelo/random_matrix.txt", "r");
	// copy matrix in BOLD
	for(size_t k = 0; k < N; k++) {
		for(size_t l = 0; l < L; l++) {
			fscanf(fp, "%f", &BOLD[k*L+l]);
		}
	}
	fclose(fp);

	printf("Copied matrix in BOLD \n");
	
	long long M11 = (N-1);
	M11 *= N;
	M11 /= 2;

	// creazione matrice triangolare superiore di dimensione (N-1) * N / 2
	float * upper_tri;
	upper_tri = (float *) malloc(M11 * sizeof(float));
	for(long long idx=0; idx<M11; idx++) {
		upper_tri[idx] = 0;
	}

	long long rem_mem = remaining_mem(N, L);
	
	if(N <= rem_mem) {
		printf("CorMat_2 \n");
		printf("Computing correlations ... \n");
		// inizio calcolo tempo di esecuzione
		start_time = omp_get_wtime(); 
		// calcolo della matrice triangolare superiore
		CorMat_2(BOLD, upper_tri, N, L);
		stop_time = omp_get_wtime(); 
		printf("Running time for computing correlations: %f \n", stop_time - start_time);
	}

	if(N > rem_mem) {
		printf("CorMat_3 \n");
		printf("Computing correlations ... \n");
		// inizio calcolo tempo di esecuzione
		start_time = omp_get_wtime(); 
		// calcolo della matrice triangolare superiore
		CorMat_3(BOLD, upper_tri, N, L, rem_mem);
		stop_time = omp_get_wtime(); 
		printf("Running time for computing correlations: %f \n", stop_time - start_time);
	}

	printf("Writing correlation values into the text file ... \n");
	fp = fopen("/home/carlo/Documents/progetto-calcolo-parallelo/openmp_pcc_corrs_TEST.txt", "w");
	for(long long idx=0; idx<M11; idx++) {
			fprintf(fp, "%f \n", upper_tri[idx]);
	}
	fclose(fp);

	free(upper_tri);
	free(BOLD);

}