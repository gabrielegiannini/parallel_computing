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

namespace fs = std::__fs::filesystem;

using namespace std;

#define DEFAULT_CLUSTER_NUMBER 5
#define ARRAYSIZEOF(ptr) (sizeof(ptr)/sizeof(ptr[0]))

void normA(const double vect[], double **centroids, double **res, double **sum, const size_t dataSize, const size_t clusterNumber){
    /* 
       Calcoliamo la norma fra un vettore e un centroide
       allora, res contiene i risultati intermedi del calcolo della norma, ovvero i quadrati delle differenze fra coordinate corrispondenti dei vettori
    */

    //res = new double[dataVec.size()][cluster_number];
    //vect e' double data[dataVec.size()];
    //centroids e' centroids = new double [cluster_number][n];
    for(int i=0 ; i<dataSize ; i++){
        for(int j=0 ; j<clusterNumber ; j++){
            res[i][j] = pow(vect[i] - centroids[j][i], 2); //non sono molto sicuro di questa riga
        }
    }

    //sum e' sum = new double[element_count][cluster_number];
    for(int k=0 ; k<dataSize ; k++){
        for(int h=0 ; h<clusterNumber ; h++){
            sum[k][h] = 0;
        }
    }
    for (int i = 0; i < dataSize; i++){
        for(int j=0; j<clusterNumber ; j++){
            sum[i][j] = sum[i][j] + res[i][j];
        }
    }
}

void meanz(double **centroids, const double data[], const int S[], const int dimS[], size_t n, size_t clusterNumber, int dataSize){
    // calcola centroidi
    //centroids e' centroids = new double [cluster_number][n];

    //set to 0 all the values
    for(int i=0 ; i<clusterNumber ; i++){
        for(int j=0 ; j<n ; j++){
            centroids[i][j] = 0;
        }
    }
    size_t dimSum = 0;

    // scorre tutti gli elementi del cluster (la grandezza del cluster e' in dimS[blockIdx.x])
    for (int i = 0; i < clusterNumber; i++){
        for (int j=0; j<n ; j++) {
            // quindi alla fine in centroids c'e' la somma di tutte le n-esime coordinate di ogni elemento del cluster
            centroids[i][j] = centroids[i][j] + data[S[j]];
            dimSum += 1;
        }
    }

    // divide per la dimensione del cluster per fare la media -> coordinata n-esima del nuovo centroide di questo cluster
    //dimS e' di dimensione clusterNumber
    for (int i = 0; i < clusterNumber; i++){
        for (int j=0; j<n ; j++) {
            centroids[i][j] = centroids[i][j] / dimS[i];
        }
    }

}


// dataSize è il numero di vettori, ovvero sizeof(data) / n (sennò aveva davvero poco senso)
void kmeanDevice(int S[], int dimS[], size_t n, double totalNormAvg[], const double data[], double **centroids, double **res,
            double **sum, size_t dataSize, uint clusterNumber)
{
    int *posMin = new int[dataSize];
    auto *min = new double[dataSize]; //inizializzare a DBL_MAX


    // array delle norme. no cuda
    for (int h = 0; h < dataSize; h++)
    {
        min[h] = DBL_MAX;
        posMin[h] = 0;
    }


    // array delle norme. no cuda
    int *filledS = new int[clusterNumber];
    for (int h = 0; h < clusterNumber; h++)
    {
        dimS[h + clusterNumber] = 0;
        totalNormAvg[h + clusterNumber] = 0;
        filledS[h] = 0;
    }

    normA(data, centroids, res, sum, dataSize, clusterNumber);

    //min sum evaluation and increase dimS in the position relative to the min value evaluated
    for (int v = 0; v < dataSize; v++){
        for (int h = 0; h < clusterNumber; h++){
            if (sum[v][h] < min[v]){
                min[v] = sum[v][h];
                posMin[v] = h;
            }
        }
        dimS[posMin[v] + clusterNumber] += 1;
    }

    //filling S
    for (int l = 0; l < dataSize; l++){
        int targetPosition = 0;
        for (int i = 0; i < posMin[l]; i++){
            targetPosition += dimS[i + clusterNumber];
        }
        targetPosition += filledS[posMin[l]];
        S[targetPosition + dataSize] = l;
        filledS[posMin[l]] += 1;
        totalNormAvg[posMin[l] + clusterNumber] =
                totalNormAvg[posMin[l] + clusterNumber] + min[l];
    }

    //evaluation of totalNormAvg
    for (int i = 0; i < clusterNumber; i++){
        if (dimS[i + clusterNumber] > 0){
            totalNormAvg[i + clusterNumber] = totalNormAvg[i + clusterNumber] / dimS[i + clusterNumber];
        }
    }

    meanz(centroids, data, S, dimS, n, clusterNumber, dataSize);
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

void initClusters(int cluster_number, unsigned long n, const double *data, double **centroidInit, size_t element_count)
{//SEQUENZIALIZZATO
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
    double **res;
    double **sum;
    int *S;
    int *S_host;
    int *S_host_old;
    int *dimS_host;
    double *totalNormAvg;
    double **centroids;
    double **centroidInit;
    double *data_d;

    int cluster_number = DEFAULT_CLUSTER_NUMBER;
    string target_file = "../../test_reale.csv";
    string output_file = "output.txt";
    bool print = false;
    int totalRuns = 100;


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
    double data[dataVec.size()];
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
    centroidInit = new double[cluster_number][n];
    int k = element_count;


    //init cluster picking random arrays from data
    srand(time(nullptr));
    initClusters(cluster_number, n, data, centroidInit, k);

    //kmeans
    size_t iterazioni = 0;
    double minAvgNorm = DBL_MAX;
    float milliseconds = 0;
    while (totalRuns > 0)
    {
        kmeanDevice(S, dimS_host, n, totalNormAvg, data_d, centroids, res, sum, element_count, cluster_number);
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