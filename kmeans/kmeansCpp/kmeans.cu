#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <iostream>
#include <string>
#include <fstream>
#include <utility>
#include <vector>
#include <unordered_map>
#include <future>
#include <filesystem>
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

void normA(const double vect[], const double centroids[], double res[], const size_t n, double sum[], const size_t dataSize,
      int kmeanIndex, const size_t clusterNumber, const int vectorsPerThread, const uint blockOffset)
{
    /* 
       Calcoliamo la norma fra un vettore e un centroide
       allora, res contiene i risultati intermedi del calcolo della norma, ovvero i quadrati delle differenze fra coordinate corrispondenti dei vettori
       quindi e' grande #vettori*#cluster*#coordinate(cioe' dimensione dei singoli vettori, cioe' n)
       
       blockIdx.y identifica il vettore di cui calcolare la norma
       blockIdx.x identifica il cluster, ovvero il centroide con cui fare la norma
       threadIdx.x identifica la coordinata di cui si deve occupare il singolo core
    */
    //printf("Indice res %lu\n",blockIdx.y*n + blockIdx.x*dataSize*n + threadIdx.x);
    // trueIndex = il vettore sul quale deve operare questo thread

    // 2a invocazione -> blockOffset = blockNum*dimensions
    const uint trueIndex = blockOffset + blockIdx.x * vectorsPerThread + threadIdx.x;
    //printf("Grappa: %lu %i %i %i %i\n", blockIdx.y * n + blockIdx.x * dataSize * n + trueIndex + kmeanIndex * dataSize * n * clusterNumber,blockIdx.x, blockIdx.y,trueIndex,kmeanIndex );
    res[trueIndex * n + blockIdx.y * dataSize * n + threadIdx.y + kmeanIndex * dataSize * n * clusterNumber] = pow(
            vect[trueIndex * n + threadIdx.y] -
            centroids[blockIdx.y * n + threadIdx.y + kmeanIndex * n * clusterNumber], 2);
    //printf("res ok \n"); 264.672
    __syncthreads();
    //__threadfence();
    if (threadIdx.y == 0)
    {
        sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] = 0;
        for (int i = 0; i < n; i++)
        {
            sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] =
                    sum[blockIdx.y * dataSize + trueIndex + kmeanIndex * dataSize * clusterNumber] +
                    res[trueIndex * n + blockIdx.y * dataSize * n + i + kmeanIndex * dataSize * n * clusterNumber];
        }
    }
}

void meanz(double centroids[], const double data[], const int S[], const int dimS[], size_t n, int kmeanIndex,
      size_t clusterNumber, int dataSize)
{// calcola centroidi
    centroids[blockIdx.x * n + threadIdx.x + kmeanIndex * n * clusterNumber] = 0;
    size_t dimSum = 0;
    // calcola la coordinata iniziale del primo vettore del cluster blockIdx.x
    for (int j = 0; j < blockIdx.x; j++)
    {
        dimSum += dimS[j + kmeanIndex * clusterNumber];
    }
//    dimSum = dimSum * n;
    // scorre tutti gli elementi del cluster (la grandezza del cluster e' in dimS[blockIdx.x])
    for (int i = 0; i < dimS[blockIdx.x + kmeanIndex * clusterNumber]; i++)
    {
        //dimSum += n;
        // quindi alla fine in centroids c'e' la somma di tutte le n-esime coordinate di ogni elemento del cluster
        centroids[blockIdx.x * n + threadIdx.x + kmeanIndex * n * clusterNumber] =
                centroids[blockIdx.x * n + threadIdx.x + kmeanIndex * n * clusterNumber] +
                data[S[dimSum + kmeanIndex * dataSize] * n + threadIdx.x];
        dimSum += 1;

    }
    // divide per la dimensione del cluster per fare la media -> coordinata n-esima del nuovo centroide di questo cluster
    centroids[blockIdx.x * n + threadIdx.x + kmeanIndex * n * clusterNumber] =
            centroids[blockIdx.x * n + threadIdx.x + kmeanIndex * n * clusterNumber] /
            dimS[blockIdx.x + kmeanIndex * clusterNumber];
}


// dataSize è il numero di vettori, ovvero sizeof(data) / n (sennò aveva davvero poco senso)
void kmeanDevice(int S[], int dimS[], size_t n, double totalNormAvg[], const double data[], double centroids[], double res[],
            double sum[], size_t dataSize, uint clusterNumber)
{
    int *posMin = new int[dataSize];
    auto *min = new double[dataSize]; //inizializzare a DBL_MAX

    for (int h = 0; h < dataSize; h++)
    {// array delle norme. no cuda
        min[h] = DBL_MAX;
        posMin[h] = 0;
    }

    int *filledS = new int[clusterNumber];
    for (int h = 0; h < clusterNumber; h++)
    {// array delle norme. no cuda
        dimS[h + clusterNumber] = 0;
        totalNormAvg[h + clusterNumber] = 0;
        filledS[h] = 0;
    }

    //norm(data, means);
    int totalThreads = clusterNumber * dataSize * n;
    // dim3 numBlocks(clusterNumber, dataSize);

    const int dimensions = __double2int_rd(1024.0 / n);
    // blocknum*n*clusternumber*dimensions ~~ totalThreads
    int blockNum = (totalThreads / (dimensions*n)) / clusterNumber;
    //printf("Sto per fare norm\n");
    dim3 blockDimensions(dimensions, n);
    dim3 gridDimension(blockNum, clusterNumber);
//    while (totalThreads > dimensions) {
//        totalThreads-=dimensions;
//        normA<<<1, blockDimensions>>>(data, centroids, res, n, sum, dataSize, threadIdx.x, clusterNumber, totalThreads);
//        cudaDeviceSynchronize();
//    }
    if (blockNum > 0)
    {
        normA<<<gridDimension, blockDimensions>>>(data, centroids, res, n, sum, dataSize, threadIdx.x, clusterNumber,
                                             dimensions, 0);
    }
    //cudaDeviceSynchronize();
    int lastVectors = dataSize - blockNum * dimensions;
    printf("===== blockNum: %i, lastVectors: %i, dimensions: %i\n", blockNum, lastVectors, dimensions);
    if (lastVectors > 0)
    {
        dim3 lastBlockDim(lastVectors, n);
        dim3 lastGridDim(1,clusterNumber);
        normA<<<lastGridDim, lastBlockDim>>>(data, centroids, res, n, sum, dataSize, threadIdx.x, clusterNumber, lastVectors,
                                   blockNum*(dimensions));
    }
    cudaDeviceSynchronize();
    for (int v = 0; v < dataSize; v++)
    {
        for (int h = 0; h < clusterNumber; h++)
        {//direi che questo for non importa parallelizzarlo con cuda visto che sono solo assegnazioni apparte norm che pero` e` gia` fatto
            if (sum[h * dataSize + clusterNumber * dataSize + v] < min[v])
            {
                min[v] = sum[h * dataSize + clusterNumber * dataSize + v];
                posMin[v] = h;
            }
        }
        dimS[posMin[v] + clusterNumber] += 1;
    }

    for (int l = 0; l < dataSize; l++)
    {
        int targetPosition = 0;
        for (int i = 0; i < posMin[l]; i++)
        {
            targetPosition += dimS[i + clusterNumber];
        }
        targetPosition += filledS[posMin[l]];
//        for (int k=0;k<n;k++){
//            S[targetPosition*n+k] = data[l*n+k];
//        }
        S[targetPosition + dataSize] = l;
        filledS[posMin[l]] += 1;
        totalNormAvg[posMin[l] + clusterNumber] =
                totalNormAvg[posMin[l] + clusterNumber] + min[l];
    }

    for (int i = 0; i < clusterNumber; i++)
    {
        if (dimS[i + clusterNumber] > 0)
        {
            totalNormAvg[i + clusterNumber] =
                    totalNormAvg[i + clusterNumber] / dimS[i + clusterNumber];
        }
    }

    meanz<<<clusterNumber, n>>>(centroids, data, S, dimS, n, threadIdx.x, clusterNumber, dataSize);
    cudaDeviceSynchronize();
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

string formatClusters(vector<string> &labels, int clusters[], const int dimS[], size_t clusterNumber, size_t dataSize)
{
    //string table = "Cluster:\n\n";
    ostringstream table;
    int width = min(max(int(labels[0].length() * 5 / 2), 6), 20);
    table << "Clusters:\n\n";
    int processedCluster[clusterNumber];
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
    size_t processed = 0;
    while (processed < dataSize)
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
    return table.str();
}

void initClusters(int cluster_number, unsigned long n, const double *data, double *centroidInit, size_t element_count)
{
    for (int i = 0; i < cluster_number; i++)
    {
        size_t randomDataPos = rand() % (element_count - 1);
//        cout << "random num: ";
//        cout << "Posizione " << i << "-esima: " << randomDataPos << endl;
        for (int j = 0; j < n; j++)
        {
            centroidInit[(i * n + j) + cluster_number * n] = data[randomDataPos * n + j];
        }
    }
}

int main(int argc, char *argv[])
{
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
    double data[dataVec.size()];
    double centroidInit[cluster_number * n];
    std::copy(dataVec.begin(), dataVec.end(), data);
    size_t element_count = dataLabel.size();
    for (string elem: dataLabel)
    {
        cout << elem << endl;
    }
    cout << "Data element number: " << element_count << "\n";
    cout << "Clusters number: " << cluster_number << "\n";
    cout << "Element dimensions (n) = " << n << endl;

    // Allocate host memory
    S_host = new int[element_count];
    S_host_old = new int[element_count];
    int *bestS = new int[element_count];
    dimS_host = new int[cluster_number];
    double *totalNormAvg_host = new double[cluster_number];

    res = new double[dataVec.size()][cluster_number];
    sum = new double[element_count][cluster_number];
    centroids = new double [cluster_number][n];

    //init cluster picking random arrays from data
    srand(time(nullptr));
    initClusters(cluster_number, n, data, centroidInit, element_count, k);

    // Executing kernel
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    size_t iterazioni = 0;
    double minAvgNorm = DBL_MAX;
    float milliseconds = 0;
    while (totalRuns > 0)
    {
        kmeanDevice(S, dimS, n, totalNormAvg, data_d, centroids, res, sum, element_count, cluster_number);

            for (int i = 0; i < element_count; i++)
            {
                if (S_host[i + k * element_count] != S_host_old[i + k * element_count])
                {
                    break;
                } else
                {
                    totalRuns--;
                    double totNorm = 0;
                    for (int h = 0; h < cluster_number; h++)
                    {
                        totNorm += totalNormAvg_host[k + h];
                    }
                    if (totNorm < minAvgNorm)
                    {
                        minAvgNorm = totNorm;
                        memcpy(bestS, S_host + k * element_count, sizeof(int) * element_count);
                    }
                }
            }
        int *tmp = S_host_old;
        S_host_old = S_host;
        S_host = tmp;
        iterazioni++;
    }


    int sommaDim = 0;
    for (int i = 0; i < cluster_number; i++)
    {
        cout << dimS_host[i] << endl;
        sommaDim+=dimS_host[i];
    }
    cout << "Totale dimS: " << sommaDim << "\n";


    string output = formatClusters(dataLabel, bestS, dimS_host, cluster_number, element_count);
    // write output on a file
    ofstream out_file;
    out_file.open(output_file);
    out_file << output << endl;
    out_file.close();
    if (print)
    {
        cout << output;
    }

    // Deallocate host memory
    delete[] S_host;
    delete[] S_host_old;
    delete[] dimS_host;
    delete[] bestS;
    delete[] totalNormAvg_host;
    delete []res;
    delete []sum;
    delete [] centroids;

    cout << "Esecuzione terminata in " << iterazioni << " iterazioni." << endl;
    cout << "" << endl;
    cout << "Tempo di esecuzione funzioni kernel: " << milliseconds / 1000 << "s" << endl;
    cout << "   -tempo di esecuzione funzione normA: " << iterazioni << " iterazioni." << endl;
    cout << "   -tempo di esecuzione funzione meanz: " << iterazioni << " iterazioni." << endl;
    cout << "" << endl;
}