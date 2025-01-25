#pragma once;

#include <algorithm>
#include <string>

#include "../src/metrics.hpp"
#include "../src/iomanager.hpp"

using namespace std;

/***
 * @author Wan-Lei Zhao
 * @date   2024-12-10
 * 
 * @copyright All rights are reserved by the author
 */

namespace cmmlab {

struct IdxItem
{

public:
    float dst;
    unsigned idx;
    IdxItem(unsigned idx0, float dst0) : idx(idx0), dst(dst0) {}

    inline friend bool operator<(const IdxItem &x, const IdxItem &y)
    {
        return x.dst < y.dst || (x.dst == y.dst && x.idx < y.idx);
    }
};

class GraphDiverse
{


public:
    void triagDiverse(std::string knnFn, std::string dataFn, std::string dstFn)
    {
        std::vector<std::vector<unsigned>> knnGraph = IOManager::loadIVECS(knnFn);
        std::vector<std::vector<unsigned>> divGraph;
        std::vector<std::vector<unsigned>> rvsGraph;
        size_t nRow = 0, nDim = 0;
        float *radius = new float[knnGraph.size()];
        float *rawDat = IOManager::loadFVECSPtr(dataFn, nRow, nDim);

        std::cout << "Graph Size: " << knnGraph.size() << std::endl;
        std::cout << "Data Size: " << nRow << "x" << nDim  << std::endl;

        for (unsigned i = 0; i < knnGraph.size(); i++)
        {
            std::vector<unsigned> nbhood1;
            divGraph.emplace_back(nbhood1);
            std::vector<unsigned> nbhood2;
            rvsGraph.emplace_back(nbhood2);
            radius[i] = RAND_MAX;
        }

        //diversify on the k-NN lists
        for (unsigned i = 0; i < knnGraph.size(); i++)
        {
            std::vector<unsigned> &nbhood = knnGraph[i];
            std::vector<unsigned> &divNb = divGraph[i];
            divNb.emplace_back(nbhood[0]);
            float *host2nbs = new float[nbhood.size()];
            unsigned hloc = nDim * i;
            for (unsigned j = 0; j < nbhood.size(); j++)
            {
                unsigned nbloc = nDim * nbhood[j];
                host2nbs[j] = Metrics::l2dst(rawDat + hloc, rawDat + nbloc, nDim);
            }
            radius[i] = host2nbs[nbhood.size() - 1];

            for (unsigned j = 1; j < nbhood.size(); j++)
            {
                bool __occlude__ = false;
                unsigned y = nbhood[j];
                for (unsigned k = 0; k < divNb.size(); k++)
                {
                    unsigned x = divNb[k];
                    float distxy = Metrics::l2dst(rawDat + x * nDim, rawDat + y * nDim, nDim);
                    if (distxy < host2nbs[j])
                    {
                        __occlude__ = true;
                        break;
                    }
                }
                if (__occlude__ == false)
                {
                    divNb.emplace_back(y);
                }
            } // for(j)
            delete[] host2nbs;
            host2nbs = nullptr;
        } //(for i)

        // collect reverse-nb graph
        for (unsigned i = 0; i < divGraph.size(); i++)
        {
            std::vector<unsigned> &divNb = divGraph[i];
            for (unsigned j = 0; j < divNb.size(); j++)
            {
                unsigned nb = divNb[j];
                std::vector<unsigned> &nbhood = rvsGraph[nb];
                float dist = Metrics::l2dst(rawDat + nb * nDim, rawDat + i * nDim, nDim);
                if (dist > radius[nb])
                {
                    nbhood.emplace_back(i);
                }
            } // for( j)
        } //(for i)

        // diversify on reverse Graph, and append to the diversified k-NN list
        for (unsigned i = 0; i < divGraph.size(); i++)
        {
            std::vector<unsigned> &divNb = divGraph[i];
            std::vector<unsigned> &rvsNb = rvsGraph[i];
            std::vector<unsigned> tmpNbs;
            for (unsigned j = 0; j < divNb.size(); j++)
            {
                tmpNbs.emplace_back(divNb[j]);
            }
            for (unsigned j = 0; j < rvsNb.size(); j++)
            {
                tmpNbs.emplace_back(rvsNb[j]);
            }
            std::vector<IdxItem> host2nbs;
            unsigned hloc = nDim * i;
            for (unsigned j = 0; j < tmpNbs.size(); j++)
            {
                unsigned nbloc = nDim * tmpNbs[j];
                float dist = Metrics::l2dst(rawDat + hloc, rawDat + nbloc, nDim);
                host2nbs.emplace_back(IdxItem(tmpNbs[j], dist));
            }
            stable_sort(host2nbs.begin(), host2nbs.end());
            unsigned nbsz = divNb.size();
            for (unsigned j = nbsz; j < host2nbs.size(); j++)
            {
                bool __occlude__ = false;
                /**
                filling your code here
                **/
               unsigned y = host2nbs[j].idx; // 获取当前邻居节点的索引
                for (unsigned k = 0; k < divNb.size(); k++)
                {
                    unsigned x = divNb[k];
                    float distxy = Metrics::l2dst(rawDat + x * nDim, rawDat + y * nDim, nDim);
                    if (distxy < host2nbs[j].dst)
                    {
                        __occlude__ = true;
                        break;
                    }
                }
                if (__occlude__ == false)
                {
                    divNb.emplace_back(y);
                    //restrict the size of the neighborhood, no larger than 64
                    if (divNb.size() >= 64)
                    {
                        break;
                    }
                }
            }
            host2nbs.clear();
        } //(for i)

        IOManager::saveIVECS(dstFn, divGraph);

        knnGraph.clear();
        rvsGraph.clear();
        divGraph.clear();
        delete[] rawDat;
        rawDat = nullptr;
        delete[] radius;
        radius = nullptr;
    }


    static void test()
    {
        std::string dataFn1 = "/home/wlzhao/datasets/bignn/sift1m/sift1m_base.fvecs";
        std::string knnFn1 = "/home/wlzhao/datasets/bignn/sift1m/sift1m_dynnd_k64.ivecs";
        std::string idxFn1 = "/home/wlzhao/datasets/bignn/sift1m/sift1m_dynnd_div_k64.ivecs";

        std::string dataFn2 = "/home/wlzhao/datasets/bignn/deep1m/deep1m_base.fvecs";
        std::string knnFn2 = "/home/wlzhao/datasets/bignn/deep1m/deep1m_nnd_k64.ivecs";
        std::string idxFn2 = "/home/wlzhao/datasets/bignn/deep1m/deep1m_xgnnd_div_k64.ivecs";

        GraphDiverse gd;
        gd.triagDiverse(knnFn1, dataFn1, idxFn1);
    }
};
}
