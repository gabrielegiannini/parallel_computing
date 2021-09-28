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
allInOne(const double data[], const double centroids[], double newCentroids[], const ulong n, double sum[],
         const ulong dataSize,
         int kmeanIndex, const ulong clusterNumber, const ulong blockOffset, int positions[])
{
    const ulong trueIndex = blockOffset + blockIdx.x * 1024 + threadIdx.x;
    int posMin = 0;
    auto min = DBL_MAX;
    for (int h = 0; h < clusterNumber; h++)
    {
        double tmpSum = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = data[trueIndex * n + i] -
                          centroids[h * n + i + kmeanIndex * n * clusterNumber];
            tmpSum = tmpSum + diff * diff;
        }
        sum[h * dataSize + kmeanIndex * clusterNumber * dataSize + trueIndex] = tmpSum;
        if (tmpSum < min)
        {
            min = tmpSum;
            posMin = h;
        }
    }
    //segna che il vettore v appartiene al cluster posMin
    positions[trueIndex] = posMin;

    //ora sappiamo a che cluster appartiene il vettore, aggiungiamolo alla somma per il calcolo del nuovo centroide di quel cluster
    for (int i = 0; i < n; i++)
    {
        atomicAdd(newCentroids + posMin * n + i, data[trueIndex * n + i]);
    }
}

// dataSize è il numero di vettori, ovvero sizeof(data) / n (sennò aveva davvero poco senso)
void
kmeanDevice(int S[], int dimS[], size_t n, double totalNormAvg[], const double data[], double centroids[],
            double sum[], size_t dataSize, uint clusterNumber, bool *convergedK,
            double *centroids_d, double newCentroids_d[], double *sum_d,
            int positions_g[], int positions_old_g[], int positions_d[])
{
    bool quit;
    quit = false;
    int *positions = positions_g;
    int *positions_old = positions_old_g;
    int *filledS = new int[clusterNumber];
    int threadidx = 0;
    uint iter = 0;
    while (!quit)
    {
        iter++;
        for (int h = 0; h < clusterNumber; h++)
        {// array delle norme. no cuda
            dimS[h + clusterNumber * threadidx] = 0;
            totalNormAvg[h + clusterNumber * threadidx] = 0;
            filledS[h] = 0;
        }

//        const uint dimensions = (uint) (1024.0l / clusterNumber);

        CUDA_CHECK_RETURN(
                cudaMemcpy(centroids_d, centroids, sizeof(double) * n * clusterNumber,
                           cudaMemcpyHostToDevice));
        fill_n(centroids, n * clusterNumber, 0.0);
        CUDA_CHECK_RETURN(
                cudaMemcpy(newCentroids_d, centroids, sizeof(double) * n * clusterNumber,
                           cudaMemcpyHostToDevice));

        uint blockNum = dataSize / 1024;
        if (blockNum > 0)
        {
            allInOne<<<blockNum, 1024>>>(data, centroids_d, newCentroids_d, n, sum_d, dataSize, 0, clusterNumber, 0,
                                         positions_d);
        }

        uint lastVectors = dataSize - blockNum * 1024;
        if (lastVectors > 0)
        {
            allInOne<<<1, lastVectors>>>(data, centroids_d, newCentroids_d, n, sum_d, dataSize, 0, clusterNumber,
                                         blockNum * 1024,
                                         positions_d);
        }

        cudaDeviceSynchronize();
        CUDA_CHECK_RETURN(cudaMemcpy(positions, positions_d, sizeof(int) * dataSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(
                cudaMemcpy(centroids, newCentroids_d, sizeof(double) * n * clusterNumber,
                           cudaMemcpyDeviceToHost));
        fill(dimS, dimS + clusterNumber, 0);
        for (int v = 0; v < dataSize; v++)
        {
            dimS[positions[v]] += 1;
        }
        for (int h = 0; h < clusterNumber; h++)
        {
            for (int d = 0; d < n; d++)
            {
                centroids[h * n + d] = centroids[h * n + d] / dimS[h];
            }
        }
        bool converged = true;
        uint k = threadidx;
        for (int i = 0; i < dataSize; i++)
        {
            if (positions[i + k * dataSize] != positions_old[i + k * dataSize])
            {
                converged = false;
                break;
            }
        }
        if (converged)
        {
            *convergedK = true;
            quit = true;

            fill(filledS, filledS + clusterNumber, 0);
            CUDA_CHECK_RETURN(
                    cudaMemcpy(sum, sum_d, sizeof(double) * dataSize * clusterNumber, cudaMemcpyDeviceToHost));
            for (int l = 0; l < dataSize; l++)
            {
                int targetPosition = 0;
                for (int i = 0; i < positions[l]; i++)
                {
                    targetPosition += dimS[i];
                }
                targetPosition += filledS[positions[l]];
                S[targetPosition + threadidx * dataSize] = l;
                filledS[positions[l]] += 1;
                totalNormAvg[positions[l] + threadidx * clusterNumber] =
                        totalNormAvg[positions[l] + threadidx * clusterNumber] + sum[positions[l] * dataSize + l];
            }

            for (int i = 0; i < clusterNumber; i++)
            {
                if (dimS[i + threadidx * clusterNumber] > 0)
                {
                    totalNormAvg[i + threadidx * clusterNumber] =
                            totalNormAvg[i + threadidx * clusterNumber] / dimS[i + threadidx * clusterNumber];
                }
            }
        }
        int *tmp = positions_old;
        positions_old = positions;
        positions = tmp;
    }

    delete[] filledS;
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
    {
        ostringstream table;

        for (size_t col = 0; col < clusterNumber; col++)
        {
            table << setw(width) << col;
            processedCluster[col] = 0;
        }
        table << setw(width / 2) << endl;
        for (int i = 0; i < clusterNumber * width; i++)
        {
            table << "·";
        }
        table << endl;
        out_file << table.str();
        if (print)
        {
            cout << table.str();
        }
    }
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
    double *sum;
    int *S_host;
    int *dimS_host;
    double *centroids;
    double *nextCentroids;
    double *data_d;
    int *positions;
    int *positions_old;
    int *positions_d;

    int cluster_number = DEFAULT_CLUSTER_NUMBER;
    string target_file = "../../test_reale.csv";
    string output_file = "output.txt";
    bool print = false;
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
    std::copy(dataVec.begin(), dataVec.end(), data);
    size_t element_count = dataLabel.size();

    // Allocate host memory
    S_host = new int[element_count];
    int *bestS = new int[element_count];
    dimS_host = new int[cluster_number];
    double *totalNormAvg_host = new double[cluster_number];
    double *sum_h = new double[element_count * cluster_number];
    positions = new int[element_count];
    positions_old = new int[element_count];
    CUDA_CHECK_RETURN(
            cudaMalloc((void **) &sum, sizeof(double) * element_count * cluster_number));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &centroids, sizeof(double) * cluster_number * n));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &nextCentroids, sizeof(double) * cluster_number * n));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &data_d, sizeof(double) * dataVec.size()));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &positions_d, sizeof(int) * element_count));

    // Transfer data from host to device memory
    CUDA_CHECK_RETURN(cudaMemcpy(data_d, data, sizeof(double) * dataVec.size(), cudaMemcpyHostToDevice));

    fill_n(centroidInit, n * cluster_number, 0.0);
    CUDA_CHECK_RETURN(cudaMemcpy(nextCentroids, centroidInit, sizeof(double) * n * cluster_number,
                                 cudaMemcpyHostToDevice));

    //init cluster picking random arrays from data
    srand(time(nullptr));
    initClusters(cluster_number, n, data, centroidInit, element_count, 0);

    CUDA_CHECK_RETURN(
            cudaMemcpy(centroids, centroidInit, sizeof(double) * n * cluster_number,
                       cudaMemcpyHostToDevice)); //i vettori inizializzati nel for prima

    size_t iterazioni = 0;
    double minAvgNorm = DBL_MAX;
    float milliseconds = 0;
    while (totalRuns > 0)
    {
        bool converged = false;
        kmeanDevice(S_host, dimS_host, n, totalNormAvg_host, data_d, centroidInit, sum_h,
                    element_count, cluster_number, &converged,
                    centroids, nextCentroids, sum, positions, positions_old, positions_d);
        float millisecondsTmp = 0;
        milliseconds = milliseconds + millisecondsTmp;

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

//    auto t2 = chrono::high_resolution_clock::now();
    formatClusters(dataLabel, bestS, dimS_host, cluster_number, element_count, print, output_file);

    cout << "Data element number: " << element_count << "\n";
    cout << "Clusters number: " << cluster_number << "\n";
    cout << "Element dimensions (n) = " << n << endl;
    cout << "Dimensione grid: " << cluster_number << "x" << element_count << endl;
//    auto elapsed = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
//    cout << "Total execution time: " << elapsed << " µs (" << elapsed / 1000000.0l << " s)." << endl;

    cout << "The elapsed time in gpu was: " << milliseconds << " ms." << endl;

    // Deallocate device memory
    cudaFree(sum);
    cudaFree(centroids);
    cudaFree(data_d);

    // Deallocate host memory
    delete[] S_host;
    delete[] dimS_host;
    delete[] bestS;
    delete[] totalNormAvg_host;

    cout << "Esecuzione terminata in " << iterazioni << " iterazioni." << endl;
    cout << "" << endl;
    cout << "Tempo di esecuzione funzioni kernel: " << milliseconds << "ms" << endl;
    cout << "" << endl;
}


