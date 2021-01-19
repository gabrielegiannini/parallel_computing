#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <future>

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

unordered_map<string, int> bigrams(string file)
{
    string b = "  ";
    b[0] = file[0];
    b[1] = file[1];
    vector<string> bigrams(1);
    bigrams[0] = b;
    unordered_map<string, int> map;
    map[b] = 1;
    for (int i = 1; i < file.length() - 2; i++)
    {
        string a = "  ";
        a[0] = file[i];
        a[1] = file[i + 1];
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
        }
        else
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
        }
        else
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
        }
        else
        {
            futArr2[p.first] = futArr1[p.first];
            futArr1[p.first] = 0;
        }
    }
    return futArr2;
}

int main()
{
    string fToString;
    fToString = fileToString("example.txt");
    future<unordered_map<string, int>> fut = async(launch::async, bigrams, fToString.substr(0, fToString.length() / 2));
    future<unordered_map<string, int>> fut2 = async(launch::async, bigrams, fToString.substr((fToString.length() / 2 - 1), (fToString.length() - fToString.length() / 2 + 1)));
    unordered_map<string, int> futArr1 = fut.get();
    unordered_map<string, int> futArr2 = fut2.get();
    unordered_map<string, int> futArr3 = mergeMap(futArr1, futArr2);
    for (pair<string, int> p : futArr3)
    {
        cout << p.first << " " << p.second << endl;
    }
    cout << endl;
    //trigrams(fToString);
}