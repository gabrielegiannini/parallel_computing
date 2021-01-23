#include <iostream>
#include <string>
#include <fstream>
#include <utility>
#include <vector>
#include <unordered_map>
#include <future>
#include <filesystem>

namespace fs = std::filesystem;

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
            futArr1[p.first] = 0;
        } else
        {
            futArr2[p.first] = futArr1[p.first];
            futArr1[p.first] = 0;
        }
    }
    return futArr2;
}

class NGramFreqComputer
{
private:
    const int n;

public:
    //make groups of n characters
    [[nodiscard]] pair<string, int> compileNgram(const string &file, int initPos) const
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

    [[nodiscard]] unordered_map<string, int> ngrams(const string &text) const
    {
        pair<string, int> p = compileNgram(text, 0);
        string b = p.first;
        vector<string> ngrams(1);
        ngrams[0] = b;
        unordered_map<string, int> map;
        map[b] = 1;
        for (int i = p.second; i < text.length() - n;)
        {
            pair<string, int> p2 = compileNgram(text, i);
            string a = p2.first;
            i = p2.second;
            bool exists = false;
            for (int j = 0; j < ngrams.size(); j++)
            {
                if (a == ngrams[j])
                {
                    exists = true;
                    break;
                }
            }
            if (!exists)
            {
                ngrams.push_back(a);
                map[a] = 1;
            } else
            {
                map[a] = map[a] + 1;
            }
        }
        return map;
    }

    explicit NGramFreqComputer(int n) : n(n)
    {}

    [[nodiscard]] int getN() const
    {
        return n;
    }
};

class FileSplitter
{
    const string &path;
    string content;
    const unsigned int splits;
    NGramFreqComputer &computer;

public:
    FileSplitter(const string &path, const unsigned splits, NGramFreqComputer &freqComputer) : path(path),
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
    }

    vector<future<unordered_map<string, int>>> splitFile()
    {
        vector<future<unordered_map<string, int>>> res(0);
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
        res.push_back(async(launch::async, [this, subFile] { return this->computer.ngrams(subFile); }));
        for (long i = splits - 1; i > 1; i--)
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
            res.push_back(async(launch::async, [this, subFile] { return this->computer.ngrams(subFile); }));
        }
        unsigned int pos = content.length() - lengthFrac - 1 + adjustment;
        while ((content[pos] & 0xC0) == 128)
        {
            pos--;
        }
        // tutto il resto del file
        subFile = content.substr(pos, 2 * lengthFrac);
        res.push_back(async(launch::async, [this, subFile] { return this->computer.ngrams(subFile); }));
        return res;
    }

    string collectCsvOutput()
    {
        vector<future<unordered_map<string, int>>> futures = splitFile();
        unordered_map<string, int> map = futures[0].get();
        for (int i = 1; i < splits; i++)
        {
            map = mergeMap(map, futures[i].get());
        }
        stringstream outString;
        outString << computer.getN() << "-gram\tOccurrencies" << endl;
        for (const auto &p : map)
        {
            outString << p.first << "\t" << p.second << endl;
        }
        return outString.str();
    }
};

int main(int argc, char *argv[])
{
    int n = 2;
    unsigned int numThreads = THREADS;
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
        } else
        {
            cerr << "opzione " << token << " non riconosciuta" << endl;
            exit(2);
        }
    }
    cout << "computing " << n << "-gram with " << numThreads << " threads." << endl;
    NGramFreqComputer computer(n);
    string fToString;
    fs::create_directory("output");
    const fs::path inputP{"analyze"};
    if (!fs::exists(inputP) || fs::is_empty(inputP))
    {
        cout << "Put all the .txt files to be analyzed in an 'analyze' directory:" << endl;
        cout << "nothing to analyze!" << endl;
    } else
    {
        for (const auto &entry: fs::directory_iterator("./analyze"))
        {
            const auto &path = entry.path();
            if (path.extension() != ".txt")
            {
                continue;
            }
            FileSplitter splitter(path, numThreads, computer);
            ofstream outFile;
            const string outPath = "analysis-" + path.stem().string() + ".csv";
            outFile.open(fs::path("output/" + outPath));
            outFile << splitter.collectCsvOutput();
        }
    }
}

