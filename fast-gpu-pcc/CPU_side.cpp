/*
 Copyright (C)Taban Eslami and Fahad Saeed
 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "memory.h"
#include <iostream>
#include <ctime>
#include <stdio.h>
#include <string.h>
#include <iomanip>
#include <fstream>
#include <stack>
#include <sstream>
#include <math.h>
using namespace std;
int CorMat_2(float* , float * , int , int );
int CorMat_3(float* , float * , int , int,long long);
long long remaining_N2(int N, int L,long long available_memory)
{
    long long x=available_memory;
    long long temp=N;
    temp*=2;
    x/=temp;
    return x;
}

// questa funzione controlla che ci sia abbastanza memoria per calcolare la matrice
long long remaining_N(int N, int L, int fflag)
{
        size_t free;
        size_t total_mem;
        cudaMemGetInfo(&free,&total_mem);
        std::cout << "Total mem:" << total_mem << std::endl;
        std::cout << "Free mem:" << free << std::endl;
        long long available_mem = free;
        available_mem/=sizeof(float);
        long long NL=N;
        NL*=L;
        if (fflag==0)
        available_mem-=(NL);
        long long x=available_mem;
        long long temp=N;
        temp*=2;
        x/=temp;
        std::cout << x << std::endl;
        return x;
}


void preprocessing(float * BOLD_t, int N,int L) {
      for (int i = 0; i < N; i++) {
        float * row = BOLD_t + i * L;
        double sum1 = 0, sum2 = 0;
        for (int l = 0; l < L; l++) {
            sum1 += row[l];
        }
        sum1 /= L;
        for (int l = 0; l < L; l++) {
            sum2 += (row[l] - sum1) * (row[l] - sum1);
        }
        sum2 = sqrt(sum2);
        for (int l = 0; l < L; l++) {
            if(sum2!=0)
                row[l] = (row[l] - sum1) / sum2;
            else
                if(sum2==0)
                row[l]=0;
        }
    }
}



int main(int argc, char *argv[])
{
    int  k = 0, l = 0;
    int N,L;
    char wr;
    N = atoi(argv[1]);
    L = atoi(argv[2]);
    wr = *(argv[3]);
    cout<<"Number of voxels: "<<N<<"  "<<"Length of time series: "<<L<<"\n\n";
    // The pseudo-random number generator is initialized using the argument passed as seed
    srand(time(0));

    // declaration processor time used by a process
    clock_t first,second;
    // riga commentata perchè prendo il file da un'altra directory
    //string name ="./";
    stringstream sstm;
    ifstream myReadFile;
    sstm.str("");
    //sstm << name<<"random_matrix.txt";
    sstm << "/home/carlo/Documents/progetto-calcolo-scientifico/random_matrix.txt";
    clock_t kho1,kho2;
    kho1=clock();
    string ad = sstm.str();
    std::cout << ad << std::endl;
    myReadFile.open(ad.c_str());
    // copy matrix in BOLD
    float * BOLD = new float [L * N];
    for(k = 0; k < N; k++) {
       for(l = 0; l < L; l++) {
            myReadFile>>BOLD[k*L+l];//BOLD[l*N+k];
        }
    }

    myReadFile.close();
    kho2=clock();
/////////////////////////////////////////////////////
    // 64 bit variable
    long long M11 = (N-1);
    M11 *= N;
    M11 /= 2;

    // declare cuda events
    cudaEvent_t start, stop;
    float time;

    // creazione matrice triangolare superiore di dimensione (N-1) * N / 2
    float * upper_tri= new float [M11];
    // inizializzazione a 0
    for(long long indii=0;indii<M11;indii++)
        upper_tri[indii]=0;
    // calcolo memoria necessaria per la matrice
    long long OOO=remaining_N(N, L, 0);

    long long temp_in=M11/2;
    long long temp_in2=M11;
    temp_in2*=3;
    temp_in2/=4;
    long long temp_in3=M11;
    temp_in3*=4;
    temp_in3/=5;


    cout<<"\nComputing correlations ..." << std::endl;

    // se la matrice occupa meno memoria del totale
    if(N<=OOO) {
        std::cout << "CorMat_2" << std::endl;
        // creates an event object 
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // records an event
        cudaEventRecord( start, 0 );

        // cout<<"\n This means that we have enough memory ";
        // calcolo tempo di esecuzione 
        first = clock();
        // calcolo della matrice pcc
        int u = CorMat_2(upper_tri,BOLD,N, L);

        second = clock();
        // stampa tempo tramite variabili clock_t
        cout<<"\nRunning time for computing correlations: \n"<<(double)(second-first)/CLOCKS_PER_SEC<<" \n";
        delete []BOLD;
        cudaEventRecord( stop, 0 );
        cudaEventSynchronize( stop );
        // ottenimento tempo trascorso per cuda 
        cudaEventElapsedTime( &time, start, stop );
    }

    // se la matrice occupa più memoria del totale
    if(N>OOO) {
        std::cout << "CorMat_3" << std::endl;
        // creates an event object 
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // records an event
        cudaEventRecord( start, 0 );
        // calcolo tempo di esecuzione 
        first = clock();
        // calcolo della matrice pcc
        CorMat_3(upper_tri, BOLD, N, L, OOO);
        second = clock();
        // stampa tempo tramite variabili clock_t
        cout<<"\nRunning time for computing correlations: \n"<<(double)(second-first)/CLOCKS_PER_SEC<<"\n ";
        delete []BOLD;
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        // ottenimento tempo trascorso per cuda 
        cudaEventElapsedTime(&time, start, stop);
    }

/*    // scrittura valori della matrice triangolare in file binario
    if (wr == 'b'){
        cout<<"\nWriting correlation values into the binary file ... \n";
        ofstream OutFile;
        OutFile.open("/home/carlo/Documents/progetto-calcolo-scientifico/corrs.bin", ios::binary | ios::out);
        OutFile.write((char*)upper_tri, sizeof(float)*M11);
        OutFile.close();
        cout<<"\nCorrelations are stored into the file corrs.bin \n";
    }

    // scrittura valori della matrice triangolare in file di testo
    if (wr == 't'){
        cout<<"\nWriting correlation values into the text file ... \n";
        ofstream correlations_print;
        correlations_print.open("/home/carlo/Documents/progetto-calcolo-scientifico/fast_gpu_pcc_corrs.txt");
        for(long long tab =0;tab<M11;tab++) {    
                   correlations_print << upper_tri[tab] << '\n';
        }
        correlations_print.close();
        cout<<"\nCorrelations are stored into the text file fast_gpu_pcc_corrs.txt \n";
        }
        return 0;*/
}
