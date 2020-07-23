#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>


/*long long remaining_N(int N, int L, int fflag) {

	size_t free;
	size_t total_mem;


	return x;
}

*/

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
	// preprocessing fMRI data
	preprocessing(BOLD, N, L);

	size_t i, j, k;
	float * BOLD_transpose;
	float * result;
	BOLD_transpose = (float *) malloc((L * N) * sizeof(float));
	result = (float *) malloc((N * N) * sizeof(float));
/*	printf("BOLD matrix: \n");
	for(i = 0; i < N; i++) {
		for(j = 0; j < L; j++) {
			printf("%f ", BOLD[i * L + j]);
		}
		printf("\n");
	}*/
	// get BOLD_transpose
	float temp;
	for(i = 0; i < N; i++) {
		for(j = 0; j < L; j++) {
			temp = BOLD[i * L + j];
			BOLD_transpose[j * N + i] = temp;
		}
	} 
/*	printf("BOLD transpose matrix: \n");
	for(i = 0; i < L; i++) {
		for(j = 0; j < N; j++) {
			printf("%f ", BOLD_transpose[i * N + j]);
		}
		printf("\n");
	}*/

/* 	# pragma omp parallel shared (N, L, BOLD, BOLD_transpose, result) private (i, j, k)
	{
		//int id;
		//id = omp_get_thread_num();
		//printf("Thread (%d) \n", id); 
		# pragma omp for
		for(i = 0; i < N; i++) {
			for(j = 0; j < N; j++) {
				result[i * N + j] = 0;
				for(k = 0; k < L; k++) {
					result[i * N + j] += BOLD[i * L + k] * BOLD_transpose[k * N + j];
				}	
			}
		}
	}*/
/*	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			result[i * N + j] = 0;
			for(k = 0; k < L; k++) {
				result[i * N + j] += BOLD[i * L + k] * BOLD_transpose[k * N + j];
			}	
		}
	}*/
/*	printf("Result matrix: \n");
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			printf("%f ", result[i * N + j]);	
		}
		printf("\n");
	}*/

	// get upper triangular matrix


	free(BOLD_transpose);
	free(result);
}


int main(int argc, char **argv) {

	int N, L;
	char wr;

	N = (int) strtol(argv[1], (char **)NULL, 10);
	L = (int) strtol(argv[2], (char **)NULL, 10);
	wr = *argv[3];

	printf("Number of voxels: %d, Length of time series: %d \n", N, L);

	clock_t start, stop;

	float * BOLD;
	BOLD = (float *) malloc((N* L) * sizeof(float));
	// open file to read matrix
	FILE *fp;
	fp = fopen("/home/carlo/Documents/progetto-calcolo-scientifico/random_matrix.txt", "r");
	// copy matrix in BOLD
	for(size_t k = 0; k < N; k++) {
		for(size_t l = 0; l < L; l++) {
			fscanf(fp, "%f", &BOLD[k*L+l]);
		}
	}
	fclose(fp);

/*	for(size_t k = 0; k < N; k++) {
		for(size_t l = 0; l < L; l++) {
			printf("%f ", BOLD[k*L+l]);
		}
			printf("\n");
	}*/

	long long M11 = (N-1);
	M11 *= N;
	M11 /= 2;

	// creazione matrice triangolare superiore di dimensione (N-1) * N / 2
	float * upper_tri;
	upper_tri = (float *) malloc(M11 * sizeof(float));
	for(long long idx=0; idx<M11; idx++) {
		upper_tri[idx] = 0;
	}
	// calcolo memoria necessaria per la matrice
	//long long OOO = remaining_N(N, L, 0);

	printf("Computing correlations ... \n");

	// inizio calcolo tempo di esecuzione
	start = clock();
	// calcolo della matrice triangolare superiore
	CorMat_2(BOLD, upper_tri, N, L);
	stop = clock();
	printf("Running time for computing correlations: %f \n", (float)(stop - start)/CLOCKS_PER_SEC);


	free(upper_tri);
	free(BOLD);

}