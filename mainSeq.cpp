#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <filesystem>

namespace fs = std::filesystem;

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
    return pair<string, int>(b, charLen + initPos);
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
    return pair<string, int>(b, wordLen + initPos);
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

unordered_map<string, int> mergeMap(unordered_map<string, int> futArr1, unordered_map<string, int> futArr2)
{
    for (const auto &p : futArr1)
    {
        if (futArr2.find(p.first) != futArr2.end())
        {
            futArr2[p.first] = futArr2[p.first] + futArr1[p.first];
//            futArr1[p.first] = 0;
        } else
        {
            futArr2[p.first] = futArr1[p.first];
//            futArr1[p.first] = 0;
        }
    }
    return futArr2;
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
    int n = 2;
    bool isNgram = true;
    /* si può passare al programma il numero di thread da avviare, default 4, oopure "hw" per indicare che deve
     * usare un num di thread in base all'hardware del pc (numThread = n° di thread della cpu)
     * si può passare anche n, ovvero la grandezza degli n-grammi da calcolare
     */
    for (int j = 1; j < argc; j++)
    {
        const string token = string(argv[j]);
        j++;
        if (token == "-n")
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
    cout << "computing " << n << (isNgram ? "-gram" : "-word") << " sequentially." << endl;
    string fToString;
    fs::create_directory("output");
    const fs::path inputP{"analyze"};
    if (!fs::exists(inputP) || fs::is_empty(inputP))
    {
        cout << "Put all the .txt files to be analyzed in an 'analyze' directory:" << endl;
        cout << "nothing to analyze!" << endl;
    } else
    {
        for (const auto &entry : fs::directory_iterator("./analyze"))
        {
            const auto &path = entry.path();
            if (path.extension() != ".txt")
            {
                continue;
            }
            fToString = fileToString(path);
            unordered_map<string, int> map = ngrams(n, fToString, isNgram);
            ofstream outFile;
            const string outPath = "analysis-" + path.stem().string() + ".csv";
            outFile.open(fs::path("output/" + outPath));
            outFile << n << "-gram\tOccurrencies" << endl;
            for (const auto &p : map)
            {
                outFile << p.first << "\t" << p.second << endl;
            }
        }
    }
}