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

#define CLUSTER_NUMBER 5
#define ARRAYSIZEOF(ptr) (sizeof(ptr)/sizeof(ptr[0]))

__global__ void normA(const double vect[], const double centroids[], double res[], size_t n, double sum[], size_t dataSize) {
    /* 
       Calcoliamo la norma fra un vettore e un centroide
       allora, res contiene i risultati intermedi del calcolo della norma, ovvero i quadrati delle differenze fra coordinate corrispondenti dei vettori
       quindi e' grande #vettori*#cluster*#coordinate(cioe' dimensione dei singoli vettori, cioe' n)
       
       blockIdx.y identifica il vettore di cui calcolare la norma
       blockIdx.x identifica il cluster, ovvero il centroide con cui fare la norma
       threadIdx.x identifica la coordinata di cui si deve occupare il singolo core
    */
    //printf("Indice res %lu\n",blockIdx.y*n + blockIdx.x*dataSize*n + threadIdx.x);
    res[blockIdx.y*n + blockIdx.x*dataSize*n + threadIdx.x] = pow(vect[blockIdx.y*n + threadIdx.x] - centroids[blockIdx.x*n + threadIdx.x], 2);
    __syncthreads();
    if(threadIdx.x == 0){
        for(int i=0; i<n;i++){
            sum[blockIdx.x*dataSize+blockIdx.y] = sum[blockIdx.x*dataSize+blockIdx.y] + res[blockIdx.y*n + blockIdx.x*dataSize*n + i];
        }
    }
}

// dataSize è il numero di vettori, ovvero sizeof(data) / n (sennò aveva davvero poco senso)
__global__ void kmeanDevice(double S[], int dimS[], size_t n, double totalNormAvg[],  const double data[], double centroids[], double res[], double sum[], size_t dataSize, size_t clusterNumber){
    int *posMin = new int[dataSize];
    auto *min = new double[dataSize]; //inizializzare a DBL_MAX

    for (int h = 0; h < dataSize; h++) {// array delle norme. no cuda
        min[h] = DBL_MAX;
        posMin[h] = 0;
    }

    int *filledS = new int[clusterNumber];
    for (int h = 0; h < clusterNumber; h++) {// array delle norme. no cuda
        dimS[h] = 0;
        totalNormAvg[h] = 0;
        filledS[h] = 0;
    }

    //norm(data, means);
    dim3 numBlocks(clusterNumber, dataSize);
    printf("Sto per fare norm\n");
    normA<<<numBlocks,n>>>(data, centroids, res, n, sum, dataSize);
    cudaDeviceSynchronize();
    for (int v=0; v<dataSize;v++){
        for (int h = 0; h < clusterNumber; h++) {//direi che questo for non importa parallelizzarlo con cuda visto che sono solo assegnazioni apparte norm che pero` e` gia` fatto
            if (sum[h*dataSize+v] < min[v]) {
                min[v] = sum[h*dataSize+v];
                posMin[v] = h;
            }
        }
        dimS[posMin[v]] += 1;
    }

    for (int l = 0; l<dataSize; l++){
        int targetPosition = 0;
        for (int i = 0; i < posMin[l]; i++) {
            targetPosition += dimS[i];
        }
        targetPosition += filledS[posMin[l]];
        for (int k=0;k<n;k++){
            S[targetPosition*n+k] = data[l*n+k];
        }
        filledS[posMin[l]] += 1;
        totalNormAvg[posMin[l]] = totalNormAvg[posMin[l]] + min[l];
    }

    for (int i = 0; i < clusterNumber; i++) {
        if (dimS[i] > 0) {
            totalNormAvg[i] = totalNormAvg[i] / dimS[i];
        }
    }
    delete[] filledS;
    delete[] min;
    delete[] posMin;
}

__global__ void meanz(double centroids[], const double S[], const int dimS[], size_t n) {// calcola centroidi
    centroids[blockIdx.x * n + threadIdx.x] = 0;
    size_t dimSum = 0;
    // calcola la coordinata iniziale del primo vettore del cluster blockIdx.x
    for (int j=0; j<blockIdx.x; j++) {
        dimSum += dimS[j];
    }
    dimSum = dimSum * n;
    // scorre tutti gli elementi del cluster (la grandezza del cluster e' in dimS[blockIdx.x])
    for (int i=0; i<dimS[blockIdx.x]; i++) {
        dimSum += n;
        // quindi alla fine in centroids c'e' la somma di tutte le n-esime coordinate di ogni elemento del cluster
        centroids[blockIdx.x * n + threadIdx.x] = centroids[blockIdx.x * n + threadIdx.x] + S[dimSum + threadIdx.x];

    }
    // divide per la dimensione del cluster per fare la media -> coordinata n-esima del nuovo centroide di questo cluster
    centroids[blockIdx.x * n + threadIdx.x] = centroids[blockIdx.x * n + threadIdx.x] / dimS[blockIdx.x];
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
            size_t start = row.find(delimiter) + 1;
            size_t end = row.find(delimiter, start);
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
    double *res;
    double *sum;
    double *sum_host;
    double *S;
    double *S_host;
    int *dimS;
    int *dimS_host;
    int *elemLength;
    double *totalNormAvg;
    double *centroids;
    double *data_d;

    vector<double> dataVec(0);
    //vector<double> totalNormAvg(CLUSTER_NUMBER);
    ifstream myfile;
    //myfile.open("../../datasetProva.csv");
    myfile.open("../../test_reale.csv");
    unsigned long n = parseData(myfile, dataVec);
    myfile.close();
    double data[dataVec.size()];
    std::copy(dataVec.begin(), dataVec.end(), data);
    //printf(dataVec.size());
    cout << "n = " << n << "\n";
    cout << "Datavec size: " << dataVec.size() << endl;
    cout << "Data size: " << ARRAYSIZEOF(data)/n << endl;

    // Allocate host memory
    S_host=(double*)malloc(sizeof(double) * dataVec.size());
    dimS_host=(int*)malloc(sizeof(int) * CLUSTER_NUMBER);
    sum_host = (double*)malloc(sizeof(double) * dataVec.size()/n*CLUSTER_NUMBER);

    // Allocate device memory
    cudaMalloc((void**)&res, sizeof(double) * dataVec.size()*CLUSTER_NUMBER);
    cudaMalloc((void**)&sum, sizeof(double) * dataVec.size()/n*CLUSTER_NUMBER);
    cudaMalloc((void**)&S, sizeof(double) * dataVec.size());
    cudaMalloc((void**)&dimS, sizeof(double) * CLUSTER_NUMBER);
    cudaMalloc((void**)&totalNormAvg, sizeof(double) * CLUSTER_NUMBER);
    cudaMalloc((void**)&elemLength, sizeof(int));
    cudaMalloc((void**)&centroids, sizeof(double) * CLUSTER_NUMBER*n);
    cudaMalloc((void**)&data_d, sizeof(double) * dataVec.size());

    // Transfer data from host to device memory
    cudaMemcpy(data_d, data, sizeof(double) * dataVec.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(elemLength, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(centroids, data, sizeof(double)*n*CLUSTER_NUMBER, cudaMemcpyHostToDevice); //i primi CLUSTER_NUMBER vettori di data per provare

    // Executing kernel 
    //normA<<<CLUSTER_NUMBER,5>>>(d_a, d_b, res, 5, sum);

    kmeanDevice<<<1,1>>>(S, dimS, n, totalNormAvg,  data_d, centroids, res, sum, ARRAYSIZEOF(data)/n, CLUSTER_NUMBER);
    cudaDeviceSynchronize();
//    meanz<<<CLUSTER_NUMBER, 5>>>(centroids, S, dimS, n);
//    cudaDeviceSynchronize();
//    kmeanDevice<<<1,1>>>(S, dimS, n, totalNormAvg,  data_d, centroids, res, sum, ARRAYSIZEOF(data)/n, CLUSTER_NUMBER);
//    cudaDeviceSynchronize();
    // Transfer data back to host memory
    cudaMemcpy(S_host, S, sizeof(double) * dataVec.size(), cudaMemcpyDeviceToHost);
    size_t index = 1;
    for(size_t i = index*n; i<(index+1)*n; i++){
        cout << S_host[i] << endl;
    }
    cout << "\n";
    cudaMemcpy(dimS_host, dimS, sizeof(int) * CLUSTER_NUMBER, cudaMemcpyDeviceToHost);
    for(int i = 0; i<CLUSTER_NUMBER; i++){
        cout << dimS_host[i] << endl;
    }
    cout << "\n";
    cudaMemcpy(sum_host, sum, sizeof(double) * dataVec.size()/n*CLUSTER_NUMBER, cudaMemcpyDeviceToHost);
    for(int i = 0; i<20; i++){
        cout << sum_host[i] << endl;
    }
    cout << "Dimensione grid: " << CLUSTER_NUMBER << "x" << ARRAYSIZEOF(data)/n << endl;

    // Verification

    // Deallocate device memory
    cudaFree(res);
    cudaFree(sum);
    cudaFree(S);
    cudaFree(dimS);
    cudaFree(totalNormAvg);
    cudaFree(elemLength);
    cudaFree(centroids);
    cudaFree(data_d);

    // Deallocate host memory
    free(S_host);
    free(dimS_host);
    free(sum_host);

    cout << "Esecuzione terminata." << endl;
}


