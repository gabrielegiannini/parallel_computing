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
#include <ctime>
#include <algorithm>
#include <cctype>

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
                              const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
              << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

__global__ void
normA(const double vect[], const double centroids[], double res[], const size_t n, double sum[], const size_t dataSize,
      int kmeanIndex, const size_t clusterNumber, const uint vectorsPerThread, const uint blockOffset)
{
    /* 
       Calcoliamo la norma fra un vettore e un centroide
       allora, res contiene i risultati intermedi del calcolo della norma, ovvero i quadrati delle differenze fra coordinate corrispondenti dei vettori
       quindi e' grande #vettori*#cluster*#coordinate(cioe' dimensione dei singoli vettori, cioe' n)
       
       blockIdx.y identifica il vettore di cui calcolare la norma
       blockIdx.x identifica il cluster, ovvero il centroide con cui fare la norma
       threadIdx.x identifica la coordinata di cui si deve occupare il singolo core
    */
    // trueIndex = il vettore sul quale deve operare questo thread
    const uint trueIndex = blockOffset + blockIdx.x * vectorsPerThread + threadIdx.x;
    double diff = abs(vect[trueIndex * n + threadIdx.y] -
                      centroids[blockIdx.y * n + threadIdx.y + kmeanIndex * n * clusterNumber]);
    res[trueIndex * n + blockIdx.y * dataSize * n + threadIdx.y + kmeanIndex * dataSize * n * clusterNumber] =
            diff * diff;
    __syncthreads();
//    __threadfence();
    if (threadIdx.y == 0)
    {
        double tmpSum = 0;
        //sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] = 0;
        for (int i = 0; i < n; i++)
        {
            tmpSum = tmpSum +
                     res[trueIndex * n + blockIdx.y * dataSize * n + i + kmeanIndex * dataSize * n * clusterNumber];
//            sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] =
//                    sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] +
//                    sharedRes[threadIdx.x*n + i + kmeanIndex * dataSize * n * clusterNumber];
//                    res[trueIndex * n + blockIdx.y * dataSize * n + i + kmeanIndex * dataSize * n * clusterNumber];
        }
        sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] = tmpSum;
    }
}

__global__ void
normA1(const double vect[], const double centroids[], double res[], const size_t n, double sum[], const size_t dataSize,
       int kmeanIndex, const size_t clusterNumber, const uint vectorsPerThread, const uint blockOffset)
{
    /*
       Calcoliamo la norma fra un vettore e un centroide
       allora, res contiene i risultati intermedi del calcolo della norma, ovvero i quadrati delle differenze fra coordinate corrispondenti dei vettori
       quindi e' grande #vettori*#cluster*#coordinate(cioe' dimensione dei singoli vettori, cioe' n)

       blockIdx.y identifica il vettore di cui calcolare la norma
       blockIdx.x identifica il cluster, ovvero il centroide con cui fare la norma
       threadIdx.x identifica la coordinata di cui si deve occupare il singolo core
    */
    // trueIndex = il vettore sul quale deve operare questo thread
    const uint trueIndex = blockOffset + blockIdx.x * vectorsPerThread + threadIdx.x;
    double diff = vect[trueIndex * n + threadIdx.y] -
                  centroids[blockIdx.y * n + threadIdx.y + kmeanIndex * n * clusterNumber];
    extern __shared__ double sharedRes[];
    //    res[trueIndex * n + blockIdx.y * dataSize * n + threadIdx.y + kmeanIndex * dataSize * n * clusterNumber] =
    //            diff * diff;
    //    printf("sharedRes[%lu]\n", blockIdx.x*n+ blockIdx.y * vectorsPerThread * n + threadIdx.y + kmeanIndex * dataSize * n * clusterNumber);
    sharedRes[threadIdx.x * n + threadIdx.y + kmeanIndex * dataSize * n * clusterNumber] =
            diff * diff;
    __syncthreads();
    //    __threadfence();
    if (threadIdx.y == 0)
    {
        double tmpSum = 0;
        //sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] = 0;
        for (int i = 0; i < n; i++)
        {
            tmpSum = tmpSum + sharedRes[threadIdx.x * n + i + kmeanIndex * dataSize * n * clusterNumber];
            //            sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] =
            //                    sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] +
            //                    sharedRes[threadIdx.x*n + i + kmeanIndex * dataSize * n * clusterNumber];
            //                    res[trueIndex * n + blockIdx.y * dataSize * n + i + kmeanIndex * dataSize * n * clusterNumber];
        }
        sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] = tmpSum;
    }
}

__global__ void
normA2(const double vect[], const double centroids[], double res[], const size_t n, double sum[], const size_t dataSize,
       int kmeanIndex, const size_t clusterNumber, const uint vectorsPerThread, const uint blockOffset)
{
    /*
       Calcoliamo la norma fra un vettore e un centroide
       allora, res contiene i risultati intermedi del calcolo della norma, ovvero i quadrati delle differenze fra coordinate corrispondenti dei vettori
       quindi e' grande #vettori*#cluster*#coordinate(cioe' dimensione dei singoli vettori, cioe' n)

       blockIdx.y identifica il vettore di cui calcolare la norma
       blockIdx.x identifica il cluster, ovvero il centroide con cui fare la norma
       threadIdx.x identifica la coordinata di cui si deve occupare il singolo core
    */
    // trueIndex = il vettore sul quale deve operare questo thread
    const uint trueIndex = blockOffset + blockIdx.x * vectorsPerThread + threadIdx.x;
//    __syncthreads();
    double tmpSum = 0;
    for (int i = 0; i < n; i++)
    {
        double diff = vect[trueIndex * n + i] -
                      centroids[blockIdx.y * n + i + kmeanIndex * n * clusterNumber];
        tmpSum = tmpSum + diff * diff;
    }
    sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] = tmpSum;
}

__global__ void
meanz(double centroids[], const double data[], const int S[], const int dimS[], size_t n, int kmeanIndex,
      size_t clusterNumber, int dataSize)
{// calcola centroidi
    //centroids[blockIdx.x * n + threadIdx.x + kmeanIndex * n * clusterNumber] = 0;
    size_t dimSum = 0;
    // calcola la coordinata iniziale del primo vettore del cluster blockIdx.x
    for (int j = 0; j < blockIdx.x; j++)
    {
        dimSum += dimS[j + kmeanIndex * clusterNumber];
    }
//    dimSum = dimSum * n;
    // scorre tutti gli elementi del cluster (la grandezza del cluster e' in dimS[blockIdx.x])
    double tmpMean = 0;
    for (int i = 0; i < dimS[blockIdx.x + kmeanIndex * clusterNumber]; i++)
    {
        //dimSum += n;
        // quindi alla fine in centroids c'e' la somma di tutte le n-esime coordinate di ogni elemento del cluster
//        centroids[blockIdx.x * n + threadIdx.x + kmeanIndex * n * clusterNumber] =
//                centroids[blockIdx.x * n + threadIdx.x + kmeanIndex * n * clusterNumber] +
//                data[S[dimSum + kmeanIndex * dataSize] * n + threadIdx.x];
        tmpMean = tmpMean + data[S[dimSum + kmeanIndex * dataSize] * n + threadIdx.x];
        dimSum += 1;

    }
    // divide per la dimensione del cluster per fare la media -> coordinata n-esima del nuovo centroide di questo cluster
    centroids[blockIdx.x * n + threadIdx.x + kmeanIndex * n * clusterNumber] = tmpMean /
                                                                               dimS[blockIdx.x +
                                                                                    kmeanIndex * clusterNumber];
}

void
meanz2(double centroids[], const double data[], const int S[], const int dimS[], size_t n, int kmeanIndex,
       size_t clusterNumber, int dataSize)
{// calcola centroidi
    for (int blockidx = 0; blockidx < clusterNumber; blockidx++)
    {
        for (int threadidx = 0; threadidx < n; threadidx++)
        {
            centroids[blockidx * n + threadidx + kmeanIndex * n * clusterNumber] = 0;
            size_t dimSum = 0;
            // calcola la coordinata iniziale del primo vettore del cluster blockIdx.x
            for (int j = 0; j < blockidx; j++)
            {
                dimSum += dimS[j + kmeanIndex * clusterNumber];
            }
            //    dimSum = dimSum * n;
            // scorre tutti gli elementi del cluster (la grandezza del cluster e' in dimS[blockIdx.x])
            for (int i = 0; i < dimS[blockidx + kmeanIndex * clusterNumber]; i++)
            {
                //dimSum += n;
                // quindi alla fine in centroids c'e' la somma di tutte le n-esime coordinate di ogni elemento del cluster
                centroids[blockidx * n + threadidx + kmeanIndex * n * clusterNumber] =
                        centroids[blockidx * n + threadidx + kmeanIndex * n * clusterNumber] +
                        data[S[dimSum + kmeanIndex * dataSize] * n + threadidx];
                dimSum += 1;

            }
            // divide per la dimensione del cluster per fare la media -> coordinata n-esima del nuovo centroide di questo cluster
            centroids[blockidx * n + threadidx + kmeanIndex * n * clusterNumber] =
                    centroids[blockidx * n + threadidx + kmeanIndex * n * clusterNumber] /
                    dimS[blockidx + kmeanIndex * clusterNumber];
        }
    }
}

// dataSize è il numero di vettori, ovvero sizeof(data) / n (sennò aveva davvero poco senso)
void
kmeanDevice(int S_gobal[], int dimS[], size_t n, double totalNormAvg[], const double data[], double centroids[],
            double res[],
            double sum[], size_t dataSize, uint clusterNumber, int S_old_global[], bool *convergedK,
            double *centroids_d, double *sum_d, int *dimS_d, int *S_d, const double data_h[])
{
//    __shared__ bool quit;
//    __shared__ int *S;
//    __shared__ int *S_old;
    bool quit;
    int *S;
    int *S_old;
    quit = false;
    S = S_gobal;
    S_old = S_old_global;
    int *posMin = new int[dataSize];
    auto *min = new double[dataSize]; //inizializzare a DBL_MAX
    int *filledS = new int[clusterNumber];
    int threadidx = 0;
    uint iter = 0;
    while (!quit)
    {
        iter++;
        for (int h = 0; h < dataSize; h++)
        {// array delle norme. no cuda
            min[h] = DBL_MAX;
            posMin[h] = 0;
        }

        for (int h = 0; h < clusterNumber; h++)
        {// array delle norme. no cuda
            dimS[h + clusterNumber * threadidx] = 0;
            totalNormAvg[h + clusterNumber * threadidx] = 0;
            filledS[h] = 0;
        }

        int totalThreads = clusterNumber * dataSize * n;

//        const int dimensions = __double2int_rd(1024.0 / n);
        const uint dimensions = (uint) (1024.0l / n);
        // blocknum*n*clusternumber*dimensions ~ totalThreads

        CUDA_CHECK_RETURN(
                cudaMemcpy(centroids_d, centroids, sizeof(double) * n * clusterNumber,
                           cudaMemcpyHostToDevice));

        ulong blockNum = (totalThreads / (dimensions * n)) / clusterNumber;
        dim3 blockDimensions(dimensions, n);
        dim3 gridDimension(blockNum, clusterNumber);
        if (blockNum > 0)
        {
//            normA<<<gridDimension, blockDimensions>>>(
//                    data, centroids_d, res, n, sum_d, dataSize, 0, clusterNumber, dimensions, 0);
            normA1<<<gridDimension, blockDimensions, sizeof(double) * dimensions * n>>>(
                    data, centroids_d, res, n, sum_d, dataSize, 0, clusterNumber, dimensions, 0);
//            normA2<<<gridDimension, dimensions>>>(data, centroids_d, res, n, sum_d, dataSize, 0, clusterNumber,
//                                                  dimensions, 0);
        }
//        cudaDeviceSynchronize();

//        cout << "\n######## TERMINATA \n" << endl;

        ulong lastVectors = dataSize - blockNum * dimensions;
        if (lastVectors > 0)
        {
            dim3 lastBlockDim(lastVectors, n);
            dim3 lastGridDim(1, clusterNumber);
//            normA<<<lastGridDim, lastBlockDim>>>(data, centroids_d, res, n, sum_d,
//                                                                                   dataSize, 0, clusterNumber,
//                                                                                   lastVectors,
//                                                                                   blockNum * (dimensions));
            normA1<<<lastGridDim, lastBlockDim, sizeof(double) * lastVectors * n>>>(data, centroids_d, res, n, sum_d,
                                                                                    dataSize, 0, clusterNumber,
                                                                                    lastVectors,
                                                                                    blockNum * (dimensions));
//            normA2<<<lastGridDim, lastVectors>>>(data, centroids_d, res, n, sum_d,
//                                                 dataSize,
//                                                 0, clusterNumber,
//                                                 lastVectors,
//                                                 blockNum * (dimensions));
        }
        cudaDeviceSynchronize();
        CUDA_CHECK_RETURN(
                cudaMemcpy(sum, sum_d, sizeof(double) * dataSize * clusterNumber,
                           cudaMemcpyDeviceToHost));
        for (int v = 0; v < dataSize; v++)
        {
            for (int h = 0; h < clusterNumber; h++)
            {//direi che questo for non importa parallelizzarlo con cuda visto che sono solo assegnazioni apparte norm che pero` e` gia` fatto
                if (sum[h * dataSize + threadidx * clusterNumber * dataSize + v] < min[v])
                {
                    min[v] = sum[h * dataSize + threadidx * clusterNumber * dataSize + v];
                    posMin[v] = h;
                }
            }
            dimS[posMin[v] + threadidx * clusterNumber] += 1;
        }

        for (int l = 0; l < dataSize; l++)
        {
            int targetPosition = 0;
            for (int i = 0; i < posMin[l]; i++)
            {
                targetPosition += dimS[i + threadidx * clusterNumber];
            }
            targetPosition += filledS[posMin[l]];
            S[targetPosition + threadidx * dataSize] = l;
            filledS[posMin[l]] += 1;
            totalNormAvg[posMin[l] + threadidx * clusterNumber] =
                    totalNormAvg[posMin[l] + threadidx * clusterNumber] + min[l];
        }

        for (int i = 0; i < clusterNumber; i++)
        {
            if (dimS[i + threadidx * clusterNumber] > 0)
            {
                totalNormAvg[i + threadidx * clusterNumber] =
                        totalNormAvg[i + threadidx * clusterNumber] / dimS[i + threadidx * clusterNumber];
            }
        }

//        CUDA_CHECK_RETURN(
//                cudaMemcpy(dimS_d, dimS, sizeof(int) * clusterNumber,
//                           cudaMemcpyHostToDevice));
//        CUDA_CHECK_RETURN(
//                cudaMemcpy(S_d, S, sizeof(int) * dataSize, cudaMemcpyHostToDevice));
//        meanz<<<clusterNumber, n>>>(centroids_d, data, S_d, dimS_d, n, threadidx, clusterNumber, dataSize);
        meanz2(centroids, data_h, S, dimS, n, threadidx, clusterNumber, dataSize);
//        cudaDeviceSynchronize();

//        CUDA_CHECK_RETURN(
//                cudaMemcpy(centroids, centroids_d, sizeof(double) * n * clusterNumber,
//                           cudaMemcpyDeviceToHost));

        bool converged = true;
        uint k = threadidx;
        for (int i = 0; i < dataSize; i++)
        {
//            if(i==0){
//                printf("Primo elemento S(%p) e S_old(%p): [%i] - [%i]\n", S, S_old, S[i + k*dataSize], S_old[i+k*dataSize]);
//            }
            if (S[i + k * dataSize] != S_old[i + k * dataSize])
            {
//                printf("Primo elemento diverso S(%p) e S_old(%p): [%i] - [%i]\n", S, S_old, S[i + k*dataSize], S_old[i+k*dataSize]);
                converged = false;
                break;
            }
        }
        if (converged)
        {
            *convergedK = true;
            quit = true;
        }
//        if (threadidx == 0)
//        {
        int *tmp = S_old;
        S_old = S;
        S = tmp;
//        }
//        printf("Questa è la fine... %i\n", quit);
    }

    delete[] filledS;
    delete[] min;
    delete[] posMin;
}

bool isNullOrWhitespace(const std::string &str)
{
    return str.empty()
           || std::all_of(str.begin(), str.end(), [](char c) {
        return std::isspace(static_cast<unsigned char>(c));
    });
}

unsigned long parseData(ifstream &csv, vector<double> &data, vector<string> &labels)
{
    double *domainMax;
    unsigned long n = 0;
    int index = 0;
    while (!csv.eof())
    {
        string row;
        getline(csv, row);
        if (isNullOrWhitespace(row)) continue; // ignore blank lines
        // perche ovviamente in c++ string.split() non esiste...
        vector<string> rowArr;
        const char delimiter = ';';
        //il primo token è il label del vettore
        labels.push_back(row.substr(0, row.find(delimiter)));
        //i seguenti sono le coordinate
        size_t start = row.find(delimiter) + 1;
        size_t end = row.find(delimiter, start);
        while (end != string::npos)
        {
            rowArr.push_back(row.substr(start, end - start));
            start = end + 1; // scansa il ';'
            end = row.find(delimiter, start);
        }
        rowArr.push_back(row.substr(start));

        if (n == 0)
        {
            n = rowArr.size();
            domainMax = new double[n];
            for (int i = 0; i < n; i++)
            {
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
    for (int j = 0; j < data.size(); j++)
    {
        data[j] = data[j] / domainMax[j % n]; // normalizza i dati -> tutto è adesso fra 0 e 1
    }
    return n;
}

void formatClusters(vector<string> &labels, int clusters[], const int dimS[], size_t clusterNumber, size_t dataSize,
                    bool print, const string &output_file)
{
    int width = min(max(int(labels[0].length() * 5 / 2), 6), 20);
//    table << "Clusters:\n\n";
    ofstream out_file;
    out_file.open(output_file);
    int *processedCluster = new int[clusterNumber];
    for (size_t col = 0; col < clusterNumber; col++)
    {
//        table << setw(width) << col;
        processedCluster[col] = 0;
    }
//    table << setw(width / 2) << endl;
//    for (int i = 0; i < clusterNumber * width; i++)
//    {
//        table << "·";
//    }
//    table << endl;
    size_t processed = 0;
    const size_t chunk = 1000;
    while (processed < dataSize)
    {
        size_t start = processed;
        ostringstream table;
        while (processed < start + chunk && processed < dataSize)
        {
            for (size_t col = 0; col < clusterNumber; col++)
            {
                if (dimS[col] > processedCluster[col])
                {
                    table << setw(width) << labels[clusters[processed]];
                    processedCluster[col] += 1;
                    processed++;
                } else
                {
                    table << setw(width) << " ";
                }
            }
            table << endl;
        }
        out_file << table.str();
        if (print)
        {
            cout << table.str();
        }
    }
    delete[] processedCluster;
    out_file.close();
}

void initClusters(int cluster_number, unsigned long n, const double *data, double *centroidInit, size_t element_count,
                  int kmeanIndex)
{
    vector<size_t> used;
    for (int i = 0; i < cluster_number; i++)
    {
        size_t randomDataPos = rand() % (element_count - 1);
        while (std::find(used.begin(), used.end(), randomDataPos) != used.end())
        {
            randomDataPos = rand() % (element_count - 1);
        }
        for (int j = 0; j < n; j++)
        {
            centroidInit[(i * n + j) + kmeanIndex * cluster_number * n] = data[randomDataPos * n + j];
        }
    }
}

int main(int argc, char *argv[])
{
//    auto t1 = chrono::high_resolution_clock::now();
    double *res;
    double *sum;
    int *S;
    int *S_old;
    int *S_old_h;
    int *S_host;
    int *dimS;
    int *dimS_host;
    double *totalNormAvg;
    double *centroids;
    double *data_d;

    int cluster_number = DEFAULT_CLUSTER_NUMBER;
    string target_file = "../../test_reale.csv";
    string output_file = "output.txt";
    bool print = false;
//    int numberOfConcurrentKmeans = 5;
    int totalRuns = 100;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp("-f", argv[i]) || !strcmp("--file", argv[i]))
        {
            target_file = argv[++i];
        } else if (!strcmp("-c", argv[i]) || !strcmp("--clusters", argv[i]))
        {
            cluster_number = stoi(argv[++i]);
        } else if (!strcmp("-o", argv[i]) || !strcmp("--output", argv[i]))
        {
            output_file = argv[++i];
        } else if (!strcmp("-p", argv[i]) || !strcmp("--print", argv[i]))
        {
            print = true;
//        } else if (!strcmp("-pk", argv[i]) || !strcmp("--parallel-kmeans", argv[i]))
//        {
//            numberOfConcurrentKmeans = stoi(argv[++i]);
        } else if (!strcmp("-tr", argv[i]) || !strcmp("--total-runs", argv[i]))
        {
            totalRuns = stoi(argv[++i]);
        } else
        {
            cerr << "Unrecognized option '" << argv[i] << "' skipped\n";
        }
    }

    vector<double> dataVec(0);
    vector<string> dataLabel(0);
    ifstream myfile;
    myfile.open(target_file);
    unsigned long n = parseData(myfile, dataVec, dataLabel);
    myfile.close();
    auto data = new double[dataVec.size()];
    double centroidInit[cluster_number * n];
//    double *centroidInit;
//    CUDA_CHECK_RETURN(cudaMallocHost((double **) &centroidInit, sizeof(double)*cluster_number * n * numberOfConcurrentKmeans));
    std::copy(dataVec.begin(), dataVec.end(), data);
    size_t element_count = dataLabel.size();

    // Allocate host memory
    S_host = new int[element_count];
    S_old_h = new int[element_count];
    int *bestS = new int[element_count];
    dimS_host = new int[cluster_number];
    double *totalNormAvg_host = new double[cluster_number];
    double *sum_h = new double[element_count * cluster_number];
//    double *sum_h;
//    CUDA_CHECK_RETURN(cudaMallocHost((double **) &sum_h, sizeof(double)*element_count * cluster_number));

    // Allocate device memory
    CUDA_CHECK_RETURN(
            cudaMalloc((void **) &res, sizeof(double) * dataVec.size() * cluster_number));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **) &sum, sizeof(double) * element_count * cluster_number));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &S, sizeof(int) * element_count));
//    CUDA_CHECK_RETURN(cudaMalloc((void **) &S_old, sizeof(int) * element_count * numberOfConcurrentKmeans));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &dimS, sizeof(int) * cluster_number));
//    CUDA_CHECK_RETURN(cudaMalloc((void **) &totalNormAvg, sizeof(double) * cluster_number * numberOfConcurrentKmeans));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &centroids, sizeof(double) * cluster_number * n));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &data_d, sizeof(double) * dataVec.size()));
//    CUDA_CHECK_RETURN(cudaMalloc((void **) &convergedK_d, sizeof(bool) * numberOfConcurrentKmeans));

    // Transfer data from host to device memory
    CUDA_CHECK_RETURN(cudaMemcpy(data_d, data, sizeof(double) * dataVec.size(), cudaMemcpyHostToDevice));

    //init cluster picking random arrays from data
    srand(time(nullptr));
    initClusters(cluster_number, n, data, centroidInit, element_count, 0);

    CUDA_CHECK_RETURN(
            cudaMemcpy(centroids, centroidInit, sizeof(double) * n * cluster_number,
                       cudaMemcpyHostToDevice)); //i vettori inizializzati nel for prima

    // Executing kernel
    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);

    size_t iterazioni = 0;
    double minAvgNorm = DBL_MAX;
    float milliseconds = 0;
    while (totalRuns > 0)
    {
        bool converged = false;
//        CUDA_CHECK_RETURN(
//                cudaMemcpy(convergedK_d, convergedK, sizeof(bool) * numberOfConcurrentKmeans, cudaMemcpyHostToDevice));
//        cudaEventRecord(start);
        kmeanDevice(S_host, dimS_host, n, totalNormAvg_host, data_d, centroidInit, res, sum_h,
                    element_count, cluster_number, S_old_h, &converged,
                    centroids, sum, dimS, S, data);
//        cudaEventRecord(stop);
//        cudaEventSynchronize(stop);
        float millisecondsTmp = 0;
//        cudaEventElapsedTime(&millisecondsTmp, start, stop);
        milliseconds = milliseconds + millisecondsTmp;

//        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
//        CUDA_CHECK_RETURN(
//                cudaMemcpy(S_host, S, sizeof(int) * element_count * numberOfConcurrentKmeans, cudaMemcpyDeviceToHost));
//        CUDA_CHECK_RETURN(cudaMemcpy(totalNormAvg_host, totalNormAvg,
//                                     sizeof(double) * cluster_number * numberOfConcurrentKmeans,
//                                     cudaMemcpyDeviceToHost));
//        CUDA_CHECK_RETURN(
//                cudaMemcpy(convergedK, convergedK_d, sizeof(bool) * numberOfConcurrentKmeans, cudaMemcpyDeviceToHost));
        if (converged)
        {
            totalRuns--;
            double totNorm = 0;
            for (int h = 0; h < cluster_number; h++)
            {
                totNorm += totalNormAvg_host[h];
            }
            if (totNorm < minAvgNorm)
            {
                minAvgNorm = totNorm;
                memcpy(bestS, S_host, sizeof(int) * element_count);
                CUDA_CHECK_RETURN(cudaMemcpy(dimS_host, dimS, sizeof(int) * cluster_number,
                                             cudaMemcpyDeviceToHost));
            }
            if (totalRuns > 0)
            {
                initClusters(cluster_number, n, data, centroidInit, element_count, 0);
                CUDA_CHECK_RETURN(
                        cudaMemcpy(centroids, centroidInit,
                                   sizeof(double) * cluster_number * n, cudaMemcpyHostToDevice));

            }
        }

        iterazioni++;
    }

//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);

//    auto t2 = chrono::high_resolution_clock::now();
//    cout << "sto per formattare" << endl;
    formatClusters(dataLabel, bestS, dimS_host, cluster_number, element_count, print, output_file);
//    cout << "ho finito" << endl;
    // write output on a file
//    ofstream out_file;
//    out_file.open(output_file);
//    out_file << output << endl;
//    out_file.close();
//    if (print)
//    {
//        cout << output;
//    }

    cout << "Data element number: " << element_count << "\n";
    cout << "Clusters number: " << cluster_number << "\n";
    cout << "Element dimensions (n) = " << n << endl;
    cout << "Dimensione grid: " << cluster_number << "x" << element_count << endl;
//    auto elapsed = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
//    cout << "Total execution time: " << elapsed << " µs (" << elapsed / 1000000.0l << " s)." << endl;

    cout << "The elapsed time in gpu was: " << milliseconds << " ms." << endl;

    // Deallocate device memory
    cudaFree(res);
    cudaFree(sum);
    cudaFree(S);
    cudaFree(S_old);
    cudaFree(dimS);
    cudaFree(totalNormAvg);
    cudaFree(centroids);
    cudaFree(data_d);
//    cudaFree(convergedK_d);

    // Deallocate host memory
    delete[] S_host;
    delete[] dimS_host;
    delete[] bestS;
    delete[] totalNormAvg_host;
//    delete[] convergedK;

    cout << "Esecuzione terminata in " << iterazioni << " iterazioni." << endl;
    cout << "" << endl;
    cout << "Tempo di esecuzione funzioni kernel: " << milliseconds << "ms" << endl;
    cout << "" << endl;
}


