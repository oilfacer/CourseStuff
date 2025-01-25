
#include "../src/iomanager.hpp"
#include "../src/nnsearch.hpp"
#include "graphdiverse.hpp"

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <string>
#include <chrono>

/***
 * @author Wan-Lei Zhao
 * @date   2024-12-10
 * 
 * @copyright All rights are reserved by the author
 */

using namespace std;
using namespace cmmlab;

float getRecall(std::vector<std::vector<unsigned>> &anng, std::vector<std::vector<unsigned>> &knng, size_t checkK)
{
    size_t hit = 0;
    size_t checkN = std::min(anng.size(), knng.size());

    for (size_t i = 0; i < checkN; ++i)
    {
        auto &ann = anng[i];
        auto &knn = knng[i];
        for (size_t j = 0; j < checkK; ++j)
        {
            auto idx = ann[j];
            for (size_t l = 0; l < checkK; ++l)
            {
                auto nb = knn[l];
                if (idx == nb)
                {
                    ++hit;
                    break;
                }
            }
        }
    }
    return 1.0 * hit / (checkK * checkN);
}

void searchRecall(string datFn, string indexPath, string queryPath, string gtPath)
{
    int RecallK = 10;
    size_t qryRow = 0, qryDim = 0;
   
    vector<vector<float>> queries = IOManager::loadFVECS(queryPath);
    qryRow = queries.size();
    qryDim = queries[0].size();
    std::vector<std::vector<unsigned>> gt = IOManager::loadIVECS(gtPath);

    NNSearch mynns(indexPath, datFn);

    std::vector<size_t> search_size_small = {10, 11, 12, 13, 15, 18, 22, 26, 28, 35, 50, 60, 70, 80, 100, 128, 156, 192, 256, 298, 348, 400, 456, 512};

    size_t topk = 10;

    std::vector<std::vector<unsigned>> searched_res(qryRow);
    std::cout << qryRow << "x" << qryDim << std::endl;
    std::vector<std::pair<float, float>> result(search_size_small.size());
    /**/

    //Normally, we repeat the search for 5 rounds, to report the stable performance
    for (int it = 0; it < 5; it++)
    {
        std::cout << "-------- iter = " << it << " -------" << std::endl;
        
        for (size_t sz_i = 0; sz_i < search_size_small.size(); ++sz_i)
        {
            auto search_size = search_size_small[sz_i];
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < qryRow; ++i)
            {
                searched_res[i] = mynns.nnSearch(queries[i].data(), topk, search_size);
            }
            /**/
            auto end = std::chrono::high_resolution_clock::now();
            float QPS = (1.0 * qryRow /
                         (1.0 * std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0));
            auto recall = getRecall(searched_res, gt, RecallK);
            result[sz_i].first = std::max(result[sz_i].first, QPS);
            result[sz_i].second = std::max(result[sz_i].second, recall);
            std::cout << search_size << "," << QPS << "," << recall <<  ", " << 0 << std::endl;
            /**/
        }
    }

    std::cout << "topk,cnt_per_second,recall_at_10" << std::endl;
    for (size_t sz_i = 0; sz_i < search_size_small.size(); ++sz_i)
    {
        std::cout << search_size_small[sz_i] << "," << result[sz_i].first << "," << result[sz_i].second << ",0" << std::endl;
    }

}

void callGraphDiverse()
{
   GraphDiverse::test();
}

void help()
{
    std::cout << "nns -q queryfile -i indexfile.ivecs -gt gtfile.ivecs -c candis.fvecs \n\n";
    std::cout << "Options:\n";
    std::cout << "\t-q\tfile of queries in fvecs format\n";
    std::cout << "\t-i\tindex file in ivecs format\n";
    std::cout << "\t-gt\tground-truth file in ivecs format\n";
    std::cout << "\t-c\tcandidate vector file in fvecs format\n\n";
    std::cout << "This software is developped by Wan-Lei Zhao\n";
    return;
}

int main(int argc, char *argv[])
{

    std::string dist_func{"l2"};
    std::string indexPath{""};
    std::string queryPath{""};
    std::string datPath{""};
    std::string gtPath{""};  

    const char *required_options[4] = {"-q", "-i", "-gt", "-c"};
    int required[4] = {0, 0, 0, 0};
    
    //if you want to test/run Graph diversification uncomment the following two lines
    //otherwise, keep them commented
    //callGraphDiverse(); 
    //return 0;

    if (argc < 9)
    {
        help();
        return 0;
    }

    for (int i = 1; i < argc; i += 2)
    {
        if (strcmp(argv[i], "-q") == 0)
        {
            queryPath = argv[i + 1];
            required[0] = 1;
        }
        else if (strcmp(argv[i], "-i") == 0)
        {
            indexPath = argv[i + 1];
            required[1] = 1;
        }
        else if (strcmp(argv[i], "-gt") == 0)
        {
            gtPath = argv[i + 1];
            required[2] = 1;
        }
        else if (strcmp(argv[i], "-c") == 0)
        {
            datPath = argv[i + 1];
            required[3] = 1;
        }
    }
    bool __missed__ = false;
    for (int i = 0; i < 3; i++)
    {
        if (required[i] == 0)
        {
            std::cout << "Required option '" << required_options[i] << "' is missing!\n";
            __missed__ = true;
        }
    }
    if (__missed__)
    {
        return 0;
    }
    searchRecall(datPath, indexPath, queryPath, gtPath);

    return 0;
}
