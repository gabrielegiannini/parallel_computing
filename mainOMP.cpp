#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <future>
#include <filesystem>
#include <omp.h>
#include <chrono>
#include <cmath>

namespace fs = std::filesystem;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

#define THREADS 4

using namespace std;

string fileToString(const string &file)
{
    string s;
    string sTotal;
    ifstream myfile;
    myfile.open(file);
    while (!myfile.eof())
    {
        getline(myfile, s);
        sTotal += s + " ";
    }
    myfile.close();
    return sTotal;
}

int charLenght(char ch)
{
    int test = ch & 0xE0;
    int charLenght = 4;
    if (test < 128)
    {
        charLenght = 1;
    } else if (test < 224)
    {
        charLenght = 2;
    } else if (test < 240)
    {
        charLenght = 3;
    }
    return charLenght;
}

//make groups of n characters
pair<string, int> compileNgram(int n, const string &file, int initPos)
{
    string b;
    int charLen = charLenght(file[initPos]);
    for (int i = initPos; i < charLen + initPos; i++)
    {
        b.push_back(file[i]);
    }
    int offset = charLen;
    int newLen;
    for (int j = 1; j < n && offset + initPos < file.length(); j++)
    {
        newLen = charLenght(file[offset + initPos]);
        for (int i = offset + initPos; i < offset + newLen + initPos; i++)
        {
            b.push_back(file[i]);
        }
        offset += newLen;
    }
    return {b, charLen + initPos};
}

//make groups of n words
pair<string, int> compileNwords(int n, const string &file, int initPos)
{
    string b;
    int wordLen;
    int i = initPos;
    for (int k = 0; k < n; k++)
    {
        while (i < file.length() && file[i] != ' ' && file[i] != '\0')
        {
            b.push_back(file[i]);
            i++;
        }
        b.push_back(' ');
        if (k == 0)
        {
            wordLen = b.size();
        }
        i++; //superiamo lo spazio
    }
    return {b, wordLen + initPos};
}

unordered_map<string, int> ngrams(int n, const string &file, bool isNGram)
{
    auto functionToInvoke = isNGram ? compileNgram : compileNwords;
    pair<string, int> p = functionToInvoke(n, file, 0);
    string b = p.first;
    unordered_map<string, int> map;
    map[b] = 1;
    for (int i = p.second; i < file.length() - n;)
    {
        pair<string, int> p2 = functionToInvoke(n, file, i);
        string a = p2.first;
        i = p2.second;
        if (map.find(a) == map.end())
        {
            map[a] = 1;
        } else
        {
            map[a] = map[a] + 1;
        }
    }
    return map;
}

unordered_map<string, int> mergeMap(unordered_map<string, int> &futArr1, unordered_map<string, int> &futArr2)
{
    for (const auto &p : futArr2)
    {
        if (futArr1.find(p.first) != futArr1.end())
        {
            futArr1[p.first] = p.second + futArr1[p.first];
            //            futArr1[p.first] = 0;
        } else
        {
            futArr1[p.first] = p.second;
            //            futArr1[p.first] = 0;
        }
    }
    return futArr1;
}

vector<string> splitFile(string file, unsigned int splits)
{
    vector<string> res(0);
    int adjustment = 0;
    unsigned long lengthFrac = file.length() / splits;
    string subFile = file.substr(0, lengthFrac + 1);

    /* se il primo byte di file dopo l'ultimo incluso in subFile
     * (che è lengthFrac + 1, perché ho chiesto sottostringa da 0 lunga lengthFrac + 1, quindi va da 0 a lengthFrac)
     * inizia per 10xxxxxx (in UTF-8) allora
     * è un pezzo di un altro carattere spezzato che inizia in subFile. Riaggiungiamolo in subFile e spostiamo il
     * "cursore" di uno avanti */
    while ((file[lengthFrac + 1 + adjustment] & 0xC0) == 128)
    {
        subFile.push_back(file[lengthFrac + 1 + adjustment]);
        adjustment++;
    }
    res.push_back(subFile);
    for (long i = splits - 1; i > 1; i--)
    {
        unsigned long pos = file.length() - (i * lengthFrac) - 1 + adjustment;
        /* ora riscorriamo all'indietro per tornare all'inizio dell'ultimo carattere messo in subFile (che dovrà
         * essere anche il primo carattere del nuovo subFile perché si sovrappongono di 1 carattere)
         * */
        while ((file[pos] & 0xC0) == 128)
        {
            pos--;
        }
        subFile = file.substr(pos, lengthFrac + 1);
        adjustment = 0;
        while ((pos + lengthFrac + 1 + adjustment) < file.length() &&
               (file[pos + lengthFrac + 1 + adjustment] & 0xC0) == 128)
        {
            subFile.push_back(file[pos + lengthFrac + 1 + adjustment]);
            adjustment++;
        }
        res.push_back(subFile);
    }
    unsigned int pos = file.length() - lengthFrac - 1 + adjustment;
    while ((file[pos] & 0xC0) == 128)
    {
        pos--;
    }
    // tutto il resto del file
    subFile = file.substr(pos, 2 * lengthFrac);
    res.push_back(subFile);
    return res;
}

int main(int argc, char *argv[])
{
    long ngrams_time = 0;
    int n = 2;
    int numThreads = THREADS;
    bool isNgram = true;
    long ngrams_per_thread = 0;
    long total_ngrams_analyzed = 0;
    int numberOfExec = 100;
    double throughputMeanAllExec = 0.0;
    /* si può passare al programma il numero di thread da avviare, default 4, oopure "hw" per indicare che deve
     * usare un num di thread in base all'hardware del pc (numThread = n° di thread della cpu)
     * si può passare anche n, ovvero la grandezza degli n-grammi da calcolare
     */
    while(numberOfExec > 0) {
    auto t1 = high_resolution_clock::now();
    for (int j = 1; j < argc; j++)
    {
        const string token = string(argv[j]);
        j++;
        if (token == "-t")
        {
            try
            {
                numThreads = stoi(argv[j]);
            }
            catch (invalid_argument &ex)
            {
                if (string(argv[j]) == "hw")
                {
                    numThreads = thread::hardware_concurrency();
                } else
                {
                    cerr << "il parametro passato non è un numero valido di threads" << endl;
                    exit(1);
                }
            }
        } else if (token == "-n")
        {
            try
            {
                n = stoi(argv[j]);
            }
            catch (invalid_argument &ex)
            {
                cerr << "il parametro passato non è un numero valido" << endl;
                exit(1);
            }
        } else if (token == "-w")
        {
            isNgram = false;
        } else
        {
            cerr << "opzione " << token << " non riconosciuta" << endl;
            exit(2);
        }
    }
    cout << "computing " << n << (isNgram ? "-gram" : "-word") << " with " << numThreads << " threads." << endl;
    string fToString;
    fs::create_directory("output");
    const fs::path inputP{"analyze"};
    if (!fs::exists(inputP) || fs::is_empty(inputP))
    {
        cout << "Put all the .txt files to be analyzed in an 'analyze' directory:" << endl;
        cout << "nothing to analyze!" << endl;
    } else
    {
        //computational effort here
        omp_set_dynamic(0);
        omp_set_num_threads(numThreads);
        for (const auto &entry : fs::directory_iterator("./analyze"))
        {
            const auto &path = entry.path();
            if (path.extension() != ".txt")
            {
                continue;
            }
            fToString = fileToString(path);
            vector<string> fileSplitted = splitFile(fToString, numThreads);
            unordered_map<string, int> results[fileSplitted.size()];
            const long maxIndex = (long) (log2(fileSplitted.size()));
            unordered_map<string, int> map;

#pragma omp parallel default(none) shared(fileSplitted, results, n, isNgram, cout, maxIndex, map, numThreads)
            {
#pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < fileSplitted.size(); i++)
                {
                    results[i] = ngrams(n, fileSplitted[i], isNgram);
                }
                for (int k = 1; k < numThreads; k = k << 1)
                {
#pragma omp for schedule(dynamic, 1)
                    for (int i = 0; i < numThreads; i++)
                    {
                        if ((i ^ k) > i && (i ^ k) < numThreads)
                        {
                            mergeMap(results[i], results[i ^ k]);
                        }
                    }
                }
            }
            ofstream outFile;
            const string outPath = "analysis-" + path.stem().string() + ".csv";
            outFile.open(fs::path("output/" + outPath));
            outFile << n << "-gram\tOccurrencies" << "\n";
            for (const auto &p : results[0])
            {
                total_ngrams_analyzed += p.second;
                outFile << p.first << "\t" << p.second << "\n";
            }
        }
        auto t2 = high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        ngrams_time += ms_int.count();
    }
    throughputMeanAllExec += 1.0 / (1.0 * ngrams_time / total_ngrams_analyzed);
    numberOfExec -- ;
    cout << numberOfExec << endl;
    }
    throughputMeanAllExec /= 100;
    total_ngrams_analyzed /= 100;
    ngrams_time /= 100;
    ngrams_per_thread = total_ngrams_analyzed/numThreads; //e' il nostro stream dimension
    cout << "total_ngrams_analysìzed: "<< total_ngrams_analyzed <<endl;
    cout << "" << endl;
    cout << "Completion time ngramsOMP: " << ngrams_time << "µs" << endl;
    cout << "" << endl;
    cout << "Throughput ngramsOMP: " << throughputMeanAllExec << " operations executed in 1/Completion time" << endl;
    cout << "" << endl;
    cout << "Service time: " << 1.0 * ngrams_time/total_ngrams_analyzed << endl;
    cout << "" << endl;
    cout << "Latency: "<< 1.0 * ngrams_time/ngrams_per_thread << endl;
    cout << "" << endl;
}