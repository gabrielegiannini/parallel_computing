#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <cfloat>
#include <sstream>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

#define DEFAULT_CLUSTER_NUMBER 5
#define ARRAYSIZEOF(ptr) (sizeof(ptr)/sizeof(ptr[0]))

// dataSize è il numero di vettori, ovvero sizeof(data) / n (sennò aveva davvero poco senso)
void kmeanDevice(int S[], int dimS[], size_t n, double totalNormAvg[], double **data, double **centroids,
                 double **sum, size_t dataSize, uint clusterNumber, int norma_time, int meanz_time,
                 double **new_centroids,
                 int positions[], int positions_old[])
{
    for (int h = 0; h < clusterNumber; h++)
    {
        dimS[h] = 0;
        totalNormAvg[h] = 0;
    }

    for (int h = 0; h < clusterNumber; h++)
    {
        for (int d = 0; d < n; d++)
        {
            new_centroids[h][d] = 0;
        }
    }

    for (int i = 0; i < dataSize; i++)
    {
        int posMin = 0;
        auto min = DBL_MAX;
        for (int j = 0; j < clusterNumber; j++)
        {
            double tmpSum = 0;
            for (int k = 0; k < n; k++)
            {
                double diff = data[i][k] - centroids[j][k];
                tmpSum += diff * diff;
            }
            if (tmpSum < min)
            {
                min = tmpSum;
                posMin = j;
            }
        }
        positions[i] = posMin;
        for (int k = 0; k < n; k++)
        {
            new_centroids[posMin][k] += data[i][k];
        }
    }

    fill_n(dimS, clusterNumber, 0);
    for (int v = 0; v < dataSize; v++)
    {
        dimS[positions[v]] += 1;
    }
    for (int h = 0; h < clusterNumber; h++)
    {
        for (int d = 0; d < n; d++)
        {
            new_centroids[h][d] = new_centroids[h][d] / dimS[h];
        }
    }

    //evaluation of totalNormAvg
    for (int i = 0; i < clusterNumber; i++)
    {
        if (dimS[i] > 0)
        {
            totalNormAvg[i] = totalNormAvg[i] / dimS[i];
        }
    }
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

void initClusters(int cluster_number, unsigned long n, const double *data, double **centroidInit, size_t element_count)
{
    for (int i = 0; i < cluster_number; i++)
    {
        size_t randomDataPos = rand() % (element_count - 1);
        for (int j = 0; j < n; j++)
        {
            centroidInit[i][j] = data[randomDataPos * n + j];
        }
    }
}

int main(int argc, char *argv[])
{
    auto t1 = chrono::high_resolution_clock::now();
    double **sum;
    int *S_host;
    int *S_host_old;
    int *dimS;
    double *totalNormAvg;
    double **centroids;
    double **centroidInit;
    double **data_bidimensional;

    int cluster_number = DEFAULT_CLUSTER_NUMBER;
    string target_file = "../test_reale.csv";
    string output_file = "output.txt";
    bool print = false;
    int totalRuns = 1;


    //command line input
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


    // read file
    vector<double> dataVec(0);
    vector<string> dataLabel(0);
    ifstream myfile;
    myfile.open(target_file);
    unsigned long n = parseData(myfile, dataVec, dataLabel);
    myfile.close();
    auto data = new double[dataVec.size()];
    std::copy(dataVec.begin(), dataVec.end(), data);
    size_t element_count = dataLabel.size();
    cout << "Data element number: " << element_count << "\n";
    cout << "Clusters number: " << cluster_number << "\n";
    cout << "Element dimensions (n) = " << n << endl;


    // Allocate host memory
    S_host = new int[element_count];
    S_host_old = new int[element_count];
    auto positions_old = new int[element_count];
    auto positions = new int[element_count];
    int *bestS = new int[element_count];
    dimS = new int[cluster_number];
    double *totalNormAvg_host = new double[cluster_number];
    totalNormAvg = new double[cluster_number];

    sum = new double *[element_count];
    for (int i = 0; i < element_count; i++)
    {
        sum[i] = new double[cluster_number];
    }

    auto new_centroids = new double *[cluster_number];
    centroids = new double *[cluster_number];
    for (int i = 0; i < cluster_number; i++)
    {
        centroids[i] = new double[n];
        new_centroids[i] = new double[n];
    }

    centroidInit = new double *[cluster_number];
    for (int i = 0; i < cluster_number; i++)
    {
        centroidInit[i] = new double[n];
    }

    data_bidimensional = new double *[element_count];
    for (int i = 0; i < element_count; i++)
    {
        data_bidimensional[i] = new double[n];
    }
    for (int j = 0; j < element_count; j++)
    {
        for (int k = 0; k < n; k++)
        {
            data_bidimensional[j][k] = data[j * n + k];
        }
    }


    //init cluster picking random arrays from data
    srand(time(nullptr));
    initClusters(cluster_number, n, data, centroids, element_count); //dovrebbe essere ok
    //kmeans
    size_t iterazioni = 0;
    double minAvgNorm = DBL_MAX;
    int kmeans_device_time = 0;
    int norma_time = 0;
    int meanz_time = 0;
    while (totalRuns > 0)
    {
        bool converged = true;
//        auto t1 = high_resolution_clock::now();
        kmeanDevice(S_host, dimS, n, totalNormAvg, data_bidimensional, centroids, sum,
                    element_count, cluster_number, norma_time, meanz_time, new_centroids, positions, positions_old);
        double **temp = centroids;
        centroids = new_centroids;
        new_centroids = temp;
//        auto t2 = high_resolution_clock::now();
//        auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
//        kmeans_device_time += ms_int.count();
        for (int i = 0; i < element_count; i++)
        {
            if (positions[i] != positions_old[i])
            {
                converged = false;
                break;
            }
        }
        if (converged)
        {
            initClusters(cluster_number, n, data, centroids, element_count);
            totalRuns--;
            double totNorm = 0;
            auto filledS = new int[cluster_number];
            fill_n(filledS, cluster_number, 0);
            for (int l = 0; l < element_count; l++)
            {
                int targetPosition = 0;
                for (int i = 0; i < positions[l]; i++)
                {
                    targetPosition += dimS[i];
                }
                targetPosition += filledS[positions[l]];
                S_host[targetPosition] = l;
                filledS[positions[l]] += 1;
                totalNormAvg[positions[l]] =
                        totalNormAvg[positions[l]] + sum[positions[l]][l];
            }
            for (int h = 0; h < cluster_number; h++)
            {
                totNorm += totalNormAvg[h];
            }
            if (totNorm < minAvgNorm)
            {
                minAvgNorm = totNorm;
                memcpy(bestS, S_host, sizeof(int) * element_count);
            }
        }
        int *tmp = positions_old;
        positions_old = positions;
        positions = tmp;
        iterazioni++;
    }

    auto t2 = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();

    string output = formatClusters(dataLabel, bestS, dimS, cluster_number, element_count);
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
    delete[] dimS;
    delete[] bestS;
    delete[] totalNormAvg_host;
//    delete[]res;
    delete[]sum;
    delete[] centroids;
    delete[] new_centroids;

    cout << "Esecuzione terminata in " << iterazioni << " iterazioni." << endl;
    cout << "" << endl;
    cout << "Total execution time: " << elapsed << " µs (" << elapsed / 1000000.0l << " s)." << endl;
    cout << "Completion time kmean device: " << kmeans_device_time << "µs" << endl;
    cout << "Completion time norma: " << norma_time << "µs" << endl;
    cout << "Completion time meanz: " << meanz_time << "µs" << endl;
    cout << "Tempo altre operazioni in kmean device: " << kmeans_device_time - norma_time - meanz_time << "µs" << endl;
    cout << "" << endl;
    cout << "Throughput kmean device: " << 1.0 / kmeans_device_time << " operations executed in 1/Completion time"
         << endl;
    cout << "Throughput norma: " << 1.0 / norma_time << " operations executed in 1/Completion time" << endl;
    cout << "Throughput meanz: " << 1.0 / meanz_time << " operations executed in 1/Completion time" << endl;
    cout << "" << endl;
    cout
            << "Service time: dato che la probabilità delle funzioni kmean device, norma e meanz è sempre 1 allora sarà equivalente al completion time"
            << endl;
    cout << "" << endl;
    cout << "Latency: uguale al Service time" << endl;

}