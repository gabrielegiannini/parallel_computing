#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
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
#include <cfloat>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

using namespace std;

#define DEFAULT_CLUSTER_NUMBER 5
#define ARRAYSIZEOF(ptr) (sizeof(ptr)/sizeof(ptr[0]))
static void CheckCudaErrorAux(const char *, unsigned, const char *,
                              cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
                              const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
    << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

__global__ void normA(const double vect[], const double centroids[], double res[], const  size_t n, double sum[], const size_t dataSize) {
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
__global__ void kmeanDevice(int S[], int dimS[], size_t n, double totalNormAvg[],  const double data[], double centroids[], double res[], double sum[], size_t dataSize, size_t clusterNumber){
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
    //printf("Sto per fare norm\n");
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
//        for (int k=0;k<n;k++){
//            S[targetPosition*n+k] = data[l*n+k];
//        }
        S[targetPosition] = l;
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

__global__ void meanz(double centroids[], const double data[], const int S[], const int dimS[], size_t n) {// calcola centroidi
    centroids[blockIdx.x * n + threadIdx.x] = 0;
    size_t dimSum = 0;
    // calcola la coordinata iniziale del primo vettore del cluster blockIdx.x
    for (int j=0; j<blockIdx.x; j++) {
        dimSum += dimS[j];
    }
//    dimSum = dimSum * n;
    // scorre tutti gli elementi del cluster (la grandezza del cluster e' in dimS[blockIdx.x])
    for (int i=0; i<dimS[blockIdx.x]; i++) {
        //dimSum += n;
        // quindi alla fine in centroids c'e' la somma di tutte le n-esime coordinate di ogni elemento del cluster
        centroids[blockIdx.x * n + threadIdx.x] = centroids[blockIdx.x * n + threadIdx.x] + data[S[dimSum]*n + threadIdx.x];
        dimSum += 1;

    }
    // divide per la dimensione del cluster per fare la media -> coordinata n-esima del nuovo centroide di questo cluster
    centroids[blockIdx.x * n + threadIdx.x] = centroids[blockIdx.x * n + threadIdx.x] / dimS[blockIdx.x];
}

unsigned long parseData(ifstream &csv, vector<double> &data, vector<string> &labels) {
    double *domainMax;
    unsigned long n = 0;
    int index = 0;
    while (!csv.eof()) {
        string row;
        getline(csv, row);
        // perche ovviamente in c++ string.split() non esiste...
        vector<string> rowArr;
        const char delimiter = ';';
        //il primo token è il label del vettore
        labels.push_back(row.substr(0, row.find(delimiter)));
        //i seguenti sono le coordinate
        size_t start = row.find(delimiter) + 1;
        size_t end = row.find(delimiter, start);
        while (end != string::npos) {
            rowArr.push_back(row.substr(start, end - start));
            start = end + 1; // scansa il ';'
            end = row.find(delimiter, start);
        }
        rowArr.push_back(row.substr(start));

        if (n == 0) {
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

string formatClusters(vector<string> &labels, int clusters[], const int dimS[], size_t clusterNumber, size_t dataSize) {
    //string table = "Cluster:\n\n";
    ostringstream table;
    size_t width = min(max(labels[0].length()*5/2, 6lu), 20lu);
    table << "Clusters:\n\n";
    int processedCluster[clusterNumber];
    for (size_t col = 0; col < clusterNumber; col++) {
        table << setw(width) << col;
        processedCluster[col] = 0;
    }
    table << setw(width/2) << endl;
    for (int i = 0; i < clusterNumber*width; i++) {
        table << "·";
    }
    table << endl;
    size_t processed = 0;
    while(processed < dataSize){
        for (size_t col = 0; col < clusterNumber; col++) {
            if (dimS[col] > processedCluster[col]) {
                table << setw(width) << labels[clusters[processed]];
                processedCluster[col] += 1;
                processed++;
            } else {
                table << setw(width) << " ";
            }
        }
        table << endl;
    }
    return table.str();
}

int main(int argc, char* argv[]){
    double *res;
    double *sum;
    int *S;
    int *S_host;
    int *S_host_old;
    int *dimS;
    int *dimS_host;
    double *totalNormAvg;
    double *centroids;
    double *data_d;

    int cluster_number = DEFAULT_CLUSTER_NUMBER;
    string target_file = "../../test_reale.csv";
    string output_file = "output.txt";
    bool print = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp("-f", argv[i]) || !strcmp("--file", argv[i])) {
            target_file = argv[++i];
        } else if (!strcmp("-c", argv[i]) || !strcmp("--clusters", argv[i])) {
            cluster_number = stoi(argv[++i]);
        } else if (!strcmp("-o", argv[i]) || !strcmp("--output", argv[i])) {
            output_file = argv[++i];
        } else if (!strcmp("-p", argv[i]) || !strcmp("--print", argv[i])) {
            print = true;
        } else {
            cerr << "Unrecognized option '" << argv[i] << "' skipped\n";
        }
    }

    vector<double> dataVec(0);
    vector<string> dataLabel(0);
    //vector<double> totalNormAvg(DEFAULT_CLUSTER_NUMBER);
    ifstream myfile;
//    myfile.open("../../datasetProva.csv");
    myfile.open(target_file);
    unsigned long n = parseData(myfile, dataVec, dataLabel);
    myfile.close();
    double data[dataVec.size()];
    double centroidInit[cluster_number*n];
    std::copy(dataVec.begin(), dataVec.end(), data);
    size_t element_count = dataLabel.size();
    cout << "Data element number: " << element_count << "\n";
    cout << "Clusters number: " << cluster_number << "\n";
    cout << "Element dimensions (n) = " << n << endl;

    // Allocate host memory
    S_host = new int[element_count];
    S_host_old = new int[element_count];
    dimS_host = new int[cluster_number];

    // Allocate device memory
    CUDA_CHECK_RETURN(cudaMalloc((void**)&res, sizeof(double) * dataVec.size()*cluster_number));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&sum, sizeof(double) * element_count*cluster_number));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&S, sizeof(int) * element_count));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&dimS, sizeof(double) * cluster_number));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&totalNormAvg, sizeof(double) * cluster_number));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&centroids, sizeof(double) * cluster_number*n));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&data_d, sizeof(double) * dataVec.size()));

    // Transfer data from host to device memory
    CUDA_CHECK_RETURN(cudaMemcpy(data_d, data, sizeof(double) * dataVec.size(), cudaMemcpyHostToDevice));

    //init cluster picking random arrays from data
    for (int i=0; i < cluster_number ; i++){
        size_t randomDataPos = rand() % (element_count-1);
//        cout << "random num: ";
//        cout << randomDataPos << endl;
        for(int j=0; j<n;j++){
            centroidInit[i*n+j] = data[randomDataPos*n + j];
        }
    }
    CUDA_CHECK_RETURN(cudaMemcpy(centroids, centroidInit, sizeof(double)*n*cluster_number, cudaMemcpyHostToDevice)); //i vettori inizializzati nel for prima

    // Executing kernel
    size_t iterazioni = 0;
    bool newClusterDifferent = true;
    while(newClusterDifferent){
        kmeanDevice<<<1,1>>>(S, dimS, n, totalNormAvg,  data_d, centroids, res, sum, element_count, cluster_number);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        meanz<<<cluster_number, n>>>(centroids, data_d, S, dimS, n);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        CUDA_CHECK_RETURN(cudaMemcpy(S_host, S, sizeof(int) * element_count, cudaMemcpyDeviceToHost));
        for(int i=0;i<element_count;i++){
            if(S_host[i]!= S_host_old[i]){
                newClusterDifferent = true;
                break;
            }else{
                newClusterDifferent = false;            }
        }
        int *tmp = S_host_old;
        S_host_old = S_host;
        S_host = tmp;
        iterazioni++;
    }
    
//    kmeanDevice<<<1,1>>>(S, dimS, n, totalNormAvg,  data_d, centroids, res, sum, ARRAYSIZEOF(data)/n, DEFAULT_CLUSTER_NUMBER);
//    cudaDeviceSynchronize();
//    meanz<<<DEFAULT_CLUSTER_NUMBER, n>>>(centroids, data_d, S, dimS, n);
//    cudaDeviceSynchronize();

    // Transfer data back to host memory
    CUDA_CHECK_RETURN(cudaMemcpy(S_host, S, sizeof(int) * element_count, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(dimS_host, dimS, sizeof(int) * cluster_number, cudaMemcpyDeviceToHost));
    cout << "Dimensione grid: " << cluster_number << "x" << element_count << endl;
    cout << "Dimensioni dei cluster\n";
    for(int i = 0; i<cluster_number; i++){
        cout << dimS_host[i] << endl;
    }
    cout << "\n";


    string output = formatClusters(dataLabel, S_host, dimS_host, cluster_number, element_count);
    // write output on a file
    ofstream out_file;
    out_file.open(output_file);
    out_file << output << endl;
    out_file.close();
    if (print) {
        cout << output;
    }

    // Verification

    // Deallocate device memory
    cudaFree(res);
    cudaFree(sum);
    cudaFree(S);
    cudaFree(dimS);
    cudaFree(totalNormAvg);
    cudaFree(centroids);
    cudaFree(data_d);

    // Deallocate host memory
    delete[] S_host;
    delete[] S_host_old;
    delete[] dimS_host;

    cout << "Esecuzione terminata in " << iterazioni << " iterazioni." << endl;
}


