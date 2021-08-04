#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>
#include <string>
#include <fstream>
#include <utility>
#include <vector>
#include <unordered_map>
#include <future>
#include <filesystem>
#include <thread>
#include <float.h>

namespace fs = std::filesystem;

using namespace std;

#define N 10000000
#define MAX_ERR 1e-6

#define CLUSTER_NUMBER 2

__global__ void normA(double a[], double b[], double res[], int n, double sum[]) {
    res[blockIdx.x*n + threadIdx.x] = pow(a[blockIdx.x*n + threadIdx.x] - b[blockIdx.x*n + threadIdx.x], 2);
    __syncthreads();
    if(threadIdx.x == 0){
        for(int i=0; i<n;i++){
            sum[blockIdx.x] = sum[blockIdx.x] + res[blockIdx.x*n + i];
        }
    }
}

__global__ void meanz(double means[], double S[], int dimS[], int * elemLengthPtr) {// calcola centroidi
    int elemLength = *elemLengthPtr;
    means[blockIdx.x*elemLength + threadIdx.x] = 0;
    int dimSum = 0;
    // calcola la coordinata iniziale del primo vettore del cluster blockIdx.x
    for (int j=0; j<blockIdx.x; j++) {
        dimSum += dimS[j];
    }
    dimSum = dimSum*elemLength;
    // scorre tutti gli elementi del cluster (la grandezza del cluster e' in dimS[blockIdx.x])
    for (int i=0; i<dimS[blockIdx.x]; i++) {
        dimSum += elemLength;
        // quindi alla fine in means c'e' la somma di tutte le n-esime coordinate di ogni elemento del cluster
        means[blockIdx.x *elemLength + threadIdx.x] = means[blockIdx.x *elemLength + threadIdx.x] + S[dimSum + threadIdx.x];

    }
    // divide per la dimensione del cluster per fare la media -> coordinata n-esima del nuovo centroide di questo cluster
    means[blockIdx.x *elemLength + threadIdx.x] = means[blockIdx.x *elemLength + threadIdx.x] / dimS[blockIdx.x];
}

__global__ string getCharAlph(int i) {
     return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i-1];
}

__global__ void kmean(double totalNormAvg[], double entry[][5], double means[]) {
        double norm = 0;
        string S [][];// dimensione clusterNumber. S e' array list di array list
        /*for (int j = 0; j < S.length; j++) {
            S[j] = new ArrayList<>();
        }*/
        for (int h = 0; h < totalNormAvg.length; h++) {// array delle norme 
            totalNormAvg[h] = 0;
        }
        for (int e = 0 ; e < entry.length; e++) {
            int posMin = 0;

            double min = DBL_MAX;
            for (int h = 0; h < means.length; h++) {
                //double norm = norm(entry.getValue(), means[h]);
                if (norm < min) {
                    min = norm;
                    posMin = h;
                }
            }
            string key = getCharAlph(e);
            S[posMin][0]=key; //è sbagliato era solo per provare
            totalNormAvg[posMin] = totalNormAvg[posMin] + min;
        }
        for (int i = 0; i < totalNormAvg.length; i++) {
            if (S[i].length > 0) {
                totalNormAvg[i] = totalNormAvg[i] / S[i].length;
            }
        }
}



unsigned long parseData(ifstream &csv, vector<double> &data) {
        double *domainMax;
        unsigned long n = -1;
        int index = 0;
        while (!csv.eof()) {
            string row;
            getline(csv, row);
            //cout << row << endl;
            istringstream iss(row);
            // perche ovviamente in c++ string.split() non esiste...
            vector<string> rowArr;
            const char delimiter = ';';
            // evita il primo token, tanto è il nome del primo vettore
            int start = row.find(delimiter) + 1;
            int end = row.find(delimiter, start);
            while (end != -1) {
                rowArr.push_back(row.substr(start, end - start));
                start = end + 1; // scansa il ';'
                end = row.find(delimiter, start);
            }
            rowArr.push_back(row.substr(start));

            if (n == -1) {
                n = rowArr.size();
                domainMax = new double[n];
                for (int i = 0; i < n; i++) {
                    domainMax[i] = std::numeric_limits<double>::lowest();
                }
            }
            if (n == rowArr.size())
            {
                for (int i = 0; i < n; i++)
                {
                    data.push_back(stod(rowArr[i]));
                    if (data[index] > domainMax[i])
                    {
                        domainMax[i] = data[index];
                    }
                    index++;
                }
            }
        }
        for(int j=0; j<data.size(); j++) {
            data[j] = data[j] / domainMax[j%n]; // normalizza i dati -> tutto è adesso fra 0 e 1
        }
        return n;
    }

int main(){
    double *a, *b, *out;
    double *d_a, *d_b, *d_out;
    double *res;
    double *sum;
    double *means;
    double *S;
    int *dimS;
    int *elemLength;
    double *data_d;

    vector<double> dataVec(0);
    vector<double> totalNormAvg(CLUSTER_NUMBER);
    string s;
    ifstream myfile;
    myfile.open("../../datasetProva.csv");
    unsigned long n = parseData(myfile, dataVec);
    myfile.close();
    double data[dataVec.size()];
    std::copy(dataVec.begin(), dataVec.end(), data);
    //printf(dataVec.size());
    cout << "n = " << n << "\n";
    cout << "Data size: " << dataVec.size() << endl;

    // Allocate host memory
    a   = (double*)malloc(sizeof(double) * N);
    b   = (double*)malloc(sizeof(double) * N);
    out = (double*)malloc(sizeof(double) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&res, sizeof(double) * 10);
    cudaMalloc((void**)&sum, sizeof(double) * 2);
    cudaMalloc((void**)&means, sizeof(double) * 2);
    cudaMalloc((void**)&S, sizeof(double) * 2);
    cudaMalloc((void**)&dimS, sizeof(double) * 2);
    cudaMalloc((void**)&elemLength, sizeof(int));
    //cudaMalloc((void**)&data_d, sizeof(double) * dataVec.size());


    cudaMalloc((void**)&d_a, sizeof(double) * N);
    cudaMalloc((void**)&d_b, sizeof(double) * N);
    cudaMalloc((void**)&d_out, sizeof(double) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(double) * N, cudaMemcpyHostToDevice);

    // Executing kernel 
    normA<<<CLUSTER_NUMBER,5>>>(d_a, d_b, res, 5, sum);

    cudaDeviceSynchronize();

    meanz<<<CLUSTER_NUMBER, 5>>>(means, S, dimS, elemLength);

    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(double) * N, cudaMemcpyDeviceToHost);

    // Verification

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a);
    free(b);
    free(out);

    cout << "Esecuzione terminata." << endl;
}


