#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>


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
	clock_t start, stop;

	// preprocessing fMRI data
	start = clock();
	preprocessing(BOLD, N, L);
	stop = clock();
	printf("Running time for preprocessing: %f \n", (float)(stop - start)/CLOCKS_PER_SEC);

	BOLD_transpose = (float *) malloc((L * N) * sizeof(float));
	result = (float *) malloc((N * N) * sizeof(float));

	// get BOLD_transpose
	start = clock();
	float temp;
	for(i = 0; i < N; i++) {
		for(j = 0; j < L; j++) {
			temp = BOLD[i * L + j];
			BOLD_transpose[j * N + i] = temp;
		}
	}
	stop = clock();
	printf("Running time for transpose: %f \n", (float)(stop - start)/CLOCKS_PER_SEC);

	// matrix product
	start = clock();
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
	stop = clock();
	printf("Running time core function: %f \n", (float)(stop - start)/CLOCKS_PER_SEC);

	start = clock();
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			result[i * N + j] = 0;
			for(k = 0; k < L; k++) {
				result[i * N + j] += BOLD[i * L + k] * BOLD_transpose[k * N + j];
			}	
		}
	}
	stop = clock();
	printf("Running time core function non parallelo: %f \n", (float)(stop - start)/CLOCKS_PER_SEC);


	// get upper triangular matrix
	start = clock();
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
	stop = clock();
	printf("Running time to get upper tri: %f \n", (float)(stop - start)/CLOCKS_PER_SEC);

	// get upper triangular matrix NON PARALLELO
	start = clock();
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
	stop = clock();
	printf("Running time to get upper tri non parallelo: %f \n", (float)(stop - start)/CLOCKS_PER_SEC);

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

	long long M11 = (N-1);
	M11 *= N;
	M11 /= 2;

	// creazione matrice triangolare superiore di dimensione (N-1) * N / 2
	float * upper_tri;
	upper_tri = (float *) malloc(M11 * sizeof(float));
	for(long long idx=0; idx<M11; idx++) {
		upper_tri[idx] = 0;
	}

	printf("Computing correlations ... \n");
	// inizio calcolo tempo di esecuzione
	start = clock();
	// calcolo della matrice triangolare superiore
	CorMat_2(BOLD, upper_tri, N, L);
	stop = clock();
	printf("Running time for computing correlations: %f \n", (float)(stop - start)/CLOCKS_PER_SEC);

	printf("Writing correlation values into the text file ... \n");
	fp = fopen("/home/carlo/Documents/progetto-calcolo-scientifico/openmp_pcc_corrs.txt", "w");
	for(long long idx=0; idx<M11; idx++) {
			fprintf(fp, "%f \n", upper_tri[idx]);
	}
	fclose(fp);

	free(upper_tri);
	free(BOLD);

}