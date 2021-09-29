#include <iostream>
#include <string>
#include <fstream>
#include <utility>
#include <vector>
#include <unordered_map>
#include <future>
#include <filesystem>
#include <thread>
#include <sstream>
#include <chrono>

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

unordered_map<string, int> mergeMap(unordered_map<string, int> futArr1, unordered_map<string, int> futArr2)
{
    for (const auto &p : futArr1)
    {
        if (futArr2.find(p.first) != futArr2.end())
        {
            futArr2[p.first] = futArr2[p.first] + futArr1[p.first];
        } else
        {
            futArr2[p.first] = futArr1[p.first];
        }
    }
    return futArr2;
}

class NGramFreqComputer
{
private:
    const int n;

public:
    pair<string, int> (NGramFreqComputer::*compile)(const string &file, const int initPos) const;

    //make groups of n characters
    [[nodiscard]] pair<string, int> compileNgram(const string &file, const int initPos) const
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
        return pair<string, int>(b, charLen + initPos);
    }

    //make groups of n words
    [[nodiscard]] pair<string, int> compileNwords(const string &file, const int initPos) const
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
            i++;
        }
        return pair<string, int>(b, wordLen + initPos);
    }

    [[nodiscard]] unordered_map<string, int> ngrams(const string &text) const
    {
        pair<string, int> p = (this->*compile)(text, 0);
        string b = p.first;
        unordered_map<string, int> map;
        map[b] = 1;
        for (int i = p.second; i < text.length() - n;)
        {
            pair<string, int> p2 = (this->*compile)(text, i);
            string a = p2.first;
            i = p2.second;
            bool exists = false;
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

    explicit NGramFreqComputer(int n, const bool isNGram) : n(n)
    {
        compile = isNGram ? &NGramFreqComputer::compileNgram : &NGramFreqComputer::compileNwords;
    }

    [[nodiscard]] int getN() const
    {
        return n;
    }
};

unordered_map<string, int>
collect(NGramFreqComputer computer, const string &text, vector<future<unordered_map<string, int>>> collectable)
{
    auto map = computer.ngrams(text);
    for (auto &fut : collectable)
    {
        auto m = fut.get();
        map = mergeMap(map, m);
    }
    return map;
}

class FileSplitter
{
    const fs::path path;
    string content;
    const unsigned int splits;
    NGramFreqComputer &computer;
    future<unordered_map<string, int>> fut;

public:
    FileSplitter(const fs::path &path, const unsigned splits, NGramFreqComputer &freqComputer) : path(path),
                                                                                                 splits(splits),
                                                                                                 computer(freqComputer)
    {
        string s;
        ifstream myfile;
        myfile.open(path);
        while (!myfile.eof())
        {
            getline(myfile, s);
            content += s + " ";
        }
        myfile.close();

        fut = splitFile();
    }

    future<unordered_map<string, int>> splitFile()
    {
        vector<future<unordered_map<string, int>>> result(0);
        int adjustment = 0;
        unsigned long lengthFrac = content.length() / splits;
        string subFile = content.substr(0, lengthFrac + 1);

        /* se il primo byte di file dopo l'ultimo incluso in subFile
         * (che è lengthFrac + 1, perché ho chiesto sottostringa da 0 lunga lengthFrac + 1, quindi va da 0 a lengthFrac)
         * inizia per 10xxxxxx (in UTF-8) allora
         * è un pezzo di un altro carattere spezzato che inizia in subFile. Riaggiungiamolo in subFile e spostiamo il
         * "cursore" di uno avanti */
        while ((content[lengthFrac + 1 + adjustment] & 0xC0) == 128)
        {
            subFile.push_back(content[lengthFrac + 1 + adjustment]);
            adjustment++;
        }

        result.push_back(
                async(launch::async, collect, this->computer, subFile, vector<future<unordered_map<string, int>>>()));
        for (unsigned long i = splits - 1; i > 1; i--)
        {
            unsigned long pos = content.length() - (i * lengthFrac) - 1 + adjustment;
            /* ora riscorriamo all'indietro per tornare all'inizio dell'ultimo carattere messo in subFile (che dovrà
             * essere anche il primo carattere del nuovo subFile perché si sovrappongono di 1 carattere)
             * */
            while ((content[pos] & 0xC0) == 128)
            {
                pos--;
            }
            subFile = content.substr(pos, lengthFrac + 1);
            adjustment = 0;
            while ((pos + lengthFrac + 1 + adjustment) < content.length() &&
                   (content[pos + lengthFrac + 1 + adjustment] & 0xC0) == 128)
            {
                subFile.push_back(content[pos + lengthFrac + 1 + adjustment]);
                adjustment++;
            }
            vector<future<unordered_map<string, int>>> toWaitForReduction;
            const unsigned int size = result.size();
            unsigned int k = 1;
            unsigned int target = (i - 1) ^ k;
            while ((splits - 1 - target) < size && target > (i - 1))
            {
                toWaitForReduction.push_back(std::move(result[splits - 1 - target]));
                k = k << 1;
                target = (i - 1) ^ k;
            }
            result.push_back(async(launch::async, collect, this->computer, subFile, std::move(toWaitForReduction)));
        }
        unsigned int pos = content.length() - lengthFrac - 1 + adjustment;
        while ((content[pos] & 0xC0) == 128)
        {
            pos--;
        }
        // tutto il resto del file
        subFile = content.substr(pos, 2 * lengthFrac);
        vector<future<unordered_map<string, int>>> toWaitForReduction;
        const unsigned int size = result.size();
        unsigned int k = 1;
        while ((splits - 1 - k) < size)
        {
            toWaitForReduction.push_back(std::move(result[splits - 1 - k]));
            k = k << 1;
        }
//        result.push_back(async(launch::async, collect, this->computer, subFile, toWaitForReduction));
//        return result;
        return async(launch::async, collect, this->computer, subFile, std::move(toWaitForReduction));
    }

    string collectCsvOutput()
    {
//        vector<future<unordered_map<string, int>>> futures = splitFile();
//        future<unordered_map<string, int>> fut = splitFile();
        unordered_map<string, int> map = fut.get();
        stringstream outString;
        outString << computer.getN() << "-gram\tOccurrencies" << "\n";
        for (const auto &p : map)
        {
            outString << p.first << "\t" << p.second << "\n";
        }
        return outString.str();
    }

    const fs::path &getPath()
    {
        return path;
    }
};

int main(int argc, char *argv[])
{
    int ngrams_time = 0;
    int n = 2;
    unsigned int numThreads = THREADS;
    bool isNgram = true;
    /* si può passare al programma il numero di thread da avviare, default 4, oopure "hw" per indicare che deve
     * usare un num di thread in base all'hardware del pc (numThread = n° di thread della cpu)
     * si può passare anche n, ovvero la grandezza degli n-grammi da calcolare
     */
    for (int j = 1; j < argc; j++)
    {
        const string token = string(argv[j]);
        j++;
        if (token == "-t")
        {
            try
            {
                numThreads = stoi(argv[j]);
            } catch (invalid_argument &ex)
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
            } catch (invalid_argument &ex)
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
    const fs::path inputP{"analyze"};
    if (!fs::exists(inputP) || fs::is_empty(inputP))
    {
        cout << "Put all the .txt files to be analyzed in an 'analyze' directory:" << endl;
        cout << "nothing to analyze!" << endl;
    } else
    {
        cout << "computing " << n << "-gram with " << numThreads << " threads." << endl;
        fs::create_directory("output");
        NGramFreqComputer computer(n, isNgram);
        vector<FileSplitter> splitters;

        //computational effort here
        auto t1 = high_resolution_clock::now();
        for (const auto &entry: fs::directory_iterator("./analyze"))
        {
            const auto &path = entry.path();
            if (path.extension() != ".txt")
            {
                continue;
            }
            splitters.emplace_back(FileSplitter(path, numThreads, computer));
        }
        auto t2 = high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        ngrams_time += ms_int.count();
        for (auto &s : splitters)
        {
            const auto path = s.getPath();
            ofstream outFile;
            const string outPath = "analysis-" + path.stem().string() + ".csv";
            outFile.open(fs::path("output/" + outPath));
            outFile << s.collectCsvOutput();
        }
    }
    cout << "" << endl;
    cout << "Completion time ngrams: " << ngrams_time << "µs" << endl;
    //cout << "Completion time norma: " << 0<< "µs" <<endl;
    //cout << "Completion time meanz: " << 0<< "µs" <<endl;
    //cout << "Tempo altre operazioni in kmean device: " << 0<< "µs" <<endl;
    cout << "" << endl;
    cout << "Throughput ngrams: " << 1.0 / ngrams_time << " operations executed in 1/Completion time" << endl;
    //cout << "Throughput norma: " << 0<< " operations executed in 1/Completion time" <<endl;
    //cout << "Throughput meanz: " << 0<< " operations executed in 1/Completion time" <<endl;
    cout << "" << endl;
    cout
            << "Service time: dato che la probabilità delle funzioni kmean device, norma e meanz è sempre 1 allora sarà equivalente al completion time"
            << endl;
    cout << "" << endl;
    cout << "Latency: uguale al Service time" << endl;
    cout << "" << endl;
}

