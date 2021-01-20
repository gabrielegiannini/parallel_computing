#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <future>

#define THREADS 4

using namespace std;

string fileToString(string file)
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
    int charLenght = 0;
    if (test < 128)
    {
        charLenght = 1;
    } else if (test < 224)
    {
        charLenght = 2;
    } else if (test < 240)
    {
        charLenght = 3;
    } else
    {
        charLenght = 4;
    }
    return charLenght;
}

pair<string, int> compileBigram(string file, int initPos)
{
    string b = "";
    int test = file[initPos];
    int charLen = charLenght(file[initPos]);
    for (int i = initPos; i < charLen + initPos; i++)
    {
        b.push_back(file[i]);
    }
    int charLen2 = charLenght(file[charLen + initPos]);
    for (int i = charLen + initPos; i < charLen + charLen2 + initPos; i++)
    {
        b.push_back(file[i]);
    }
    return pair<string, int>(b, charLen + initPos);
}

unordered_map<string, int> bigrams(string file, int id)
{
    pair<string, int> p = compileBigram(file, 0);
    string b = p.first;
    vector<string> bigrams(1);
    bigrams[0] = b;
    unordered_map<string, int> map;
    map[b] = 1;
    for (int i = p.second; i < file.length() - 2;)
    {
        pair<string, int> p2 = compileBigram(file, i);
        string a = p2.first;
        i = p2.second;
        bool exists = false;
        for (int j = 0; j < bigrams.size(); j++)
        {
            if (a == bigrams[j])
            {
                exists = true;
                break;
            }
        }
        if (exists == false)
        {
            bigrams.push_back(a);
            map[a] = 1;
        } else
        {
            map[a] = map[a] + 1;
        }
    }
    return map;
}

void trigrams(string file)
{
    string b = "   ";
    b[0] = file[0];
    b[1] = file[1];
    b[2] = file[2];
    vector<string> trigrams(1);
    trigrams[0] = b;
    unordered_map<string, int> map;
    map[b] = 1;
    for (int i = 1; i < file.length() - 3; i++)
    {
        string a = "   ";
        a[0] = file[i];
        a[1] = file[i + 1];
        a[2] = file[i + 2];
        bool exists = false;
        for (int j = 0; j < trigrams.size(); j++)
        {
            if (a == trigrams[j])
            {
                exists = true;
                break;
            }
        }
        if (exists == false)
        {
            trigrams.push_back(a);
            map[a] = 1;
        } else
        {
            map[a] = map[a] + 1;
        }
    }
}

unordered_map<string, int> mergeMap(unordered_map<string, int> futArr1, unordered_map<string, int> futArr2)
{
    for (pair<string, int> p : futArr1)
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

vector<future<unordered_map<string, int>>> splitFile(string file, int splits)
{
    vector<future<unordered_map<string, int>>> res(0);
    int adjustment = 0;
    unsigned long lengthFrac = file.length() / splits;
    // il resto per aggiustare il numero di caratteri (eventuali caratteri in più finiscono nel primo subFile)
    unsigned long remainder = file.length() % splits;
    string subFile = file.substr(0, lengthFrac + 1 + remainder);

    /* se il primo byte di file dopo l'ultimo incluso in subFile
     * (che è lengthFrac + 1, perché ho chiesto sottostringa da 0 lunga lengthFrac + 1, quindi va da 0 a lengthFrac)
     * inizia per 10xxxxxx (in UTF-8) allora
     * è un pezzo di un altro carattere spezzato che inizia in subFile. Riaggiungiamolo in subFile e spostiamo il
     * "cursore" di uno avanti*/
    while ((file[lengthFrac + 1 + adjustment + remainder] & 0xC0) == 128)
    {
        subFile.push_back(file[lengthFrac + 1 + adjustment + remainder]);
        adjustment++;
    }
    res.push_back(async(launch::async, bigrams, subFile, 0));
    for (int i = splits - 1; i > 0; i--)
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
        res.push_back(async(launch::async, bigrams, subFile, i));
    }
    return res;
}

int main(int argc, char *argv[])
{
    int numThreads = THREADS;
    // si può passare al programma il numero di thread da avviare, default 4
    if (argc == 2)
    {
        try
        {
            numThreads = stoi(argv[1]);
        } catch (invalid_argument)
        {
            cerr << "il parametro passato non è un numero valido di threads" << endl;
        }
    }
    string fToString;
    fToString = fileToString("example.txt");
    vector<future<unordered_map<string, int>>> futures = splitFile(fToString, numThreads);
    unordered_map<string, int> map = futures.at(0).get();
    for (int i = 1; i < numThreads; i++)
    {
        map = mergeMap(map, futures.at(i).get());
    }
    for (pair<string, int> p : map)
    {
        cout << p.first << " " << p.second << endl;
    }
    cout << endl;
    cout << "eseguito con " << numThreads << " threads." << endl;
}