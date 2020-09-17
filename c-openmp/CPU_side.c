#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


long long remaining_mem(int N, int L) {
	long long available_mem = 16000000000;
	// division by float size = 4 byte
	available_mem /= 4;
	available_mem -= N * L;
	float temp = N * 2;
	available_mem /= temp;
	printf("available_mem: %lld \n", available_mem);
	return available_mem;
}


float remaining_N2(int N, int L, long long rem_mem) {
	long long x=rem_mem;
    long long temp=N;
    temp*=2;
    x/=temp;
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

/*	start_time = omp_get_wtime(); 
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			result[i * N + j] = 0;
			for(k = 0; k < L; k++) {
				result[i * N + j] += BOLD[i * L + k] * BOLD_transpose[k * N + j];
			}	
		}
	}
	stop_time = omp_get_wtime();
	printf("Running time core function non parallelo: %f \n", stop_time - start_time);*/


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

	// get upper triangular matrix NON PARALLELO
/*	start_time = omp_get_wtime(); 
	int idx;
	idx = 0;
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			if(i < j) {
				upper_tri[idx] = result[i * N + j];
				idx += 1;
			}
		}
	}
	stop_time = omp_get_wtime();
	printf("Running time to get upper tri non parallelo: %f \n", stop_time - start_time);*/

	free(result);
	free(BOLD_transpose);
}


void CorMat_3(float * BOLD, float* upper_tri, int N, int L, long long rem_mem) {
	
	size_t i, j, k;
	float * BOLD_transpose;
	float * result;

	double start_time, stop_time;

	// preprocessing fMRI data
	start_time = omp_get_wtime(); 
	preprocessing(BOLD, N, L);
	stop_time = omp_get_wtime();
	printf("Running time for preprocessing: %f \n", stop_time - start_time);

	int flag = 1;
	int count = 0;
	int block = 0;
	int N_prime = N;
	int so_far = 0;
	int pak = 0;
	long long M1, temp, temp2=0, temp3=0;
	long long cormat_fullsize;
	//float * upper;

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
			//free(upper);
			free(result); //devCormat in C++ CUDA
		}

		cormat_fullsize = block;
		cormat_fullsize *= N_prime;

		result = (float *) malloc(cormat_fullsize * sizeof(float));
		//upper = (float *) malloc(M1 * sizeof(float));

		pak++;

		// matrix product
		start_time = omp_get_wtime(); 
 		# pragma omp parallel shared (block, L, BOLD, BOLD_transpose, result) private (i, j, k)
		{
			# pragma omp for
			for(i = 0; i < block; i++) {
				for(j = 0; j < block; j++) {
					result[i * N + j] = 0;
					for(k = 0; k < L; k++) {
						result[i * N + j] += BOLD[i * L + k] * BOLD_transpose[k * N + j];
					}	
				}
			}
		}
		stop_time = omp_get_wtime();
		printf("Running time core function: %f \n", stop_time - start_time);

		temp2 = block;
		temp2 *= N_prime;

		// get upper triangular matrix
		start_time = omp_get_wtime(); 
		# pragma omp parallel shared (block, upper_tri, result) private (i, j)
		{
			int idx = 0;
			for(i = 0; i < block; i++) {
				for(j = 0; j < block; j++) {
					idx = (block * i) + j - ((i * (i+1)) / 2);
					idx -= (i+1);
					if(i < j) {
						upper_tri[idx] = result[i * block + j];
					}
				}
			}
		}
		stop_time = omp_get_wtime();
		printf("Running time to get upper tri: %f \n", stop_time - start_time);

		temp3 += M1;
		so_far += block;
		if(N_prime>block)
	        {
	            N_prime=N_prime-block;
	            block=remaining_N2(N_prime, L, rem_mem);
	          
	            if(N_prime  <block)//checking last chunk
	             block=N_prime;

	        }
		count++;
	}

	free(result);
	free(BOLD_transpose);


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


/*	printf("Writing correlation values into the text file ... \n");
	fp = fopen("/home/carlo/Documents/progetto-calcolo-parallelo/openmp_pcc_corrs.txt", "w");
	for(long long idx=0; idx<M11; idx++) {
			fprintf(fp, "%f \n", upper_tri[idx]);
	}
	fclose(fp);*/

	free(upper_tri);
	free(BOLD);

}