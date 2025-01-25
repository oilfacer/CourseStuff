#pragma once;

#include "iomanager.hpp"
#include "metrics.hpp"
#include <queue>
#include <assert.h>
#include <vector>
#include <string>

/***
 * @author Wan-Lei Zhao
 * @date   2024-12-10
 * 
 * @copyright All rights are reserved by the author
 */

namespace cmmlab
{
    class NNSearch
    {
        using PriorityQType =
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>>;

    private:
        std::vector<std::vector<unsigned>> nnGraph;
        float *vectDat{nullptr};
        size_t nDim{0}, nRow{0};
        vector<unsigned char> flag; // to indicate whether a node has been visited

    public:

        NNSearch(std::string graphFn, string vectFn)
        {
            this->nnGraph = IOManager::loadIVECS(graphFn);
            assert(nnGraph.size() > 0);
            this->vectDat = IOManager::loadFVECSPtr(vectFn, this->nRow, this->nDim);
            std::cout << "Data Size ............................. " << this->nRow << "x" << this->nDim << std::endl;
            std::cout << this->nnGraph.size() << std::endl;
            this->flag.resize(this->nRow + 1, 0);
        }

        inline size_t randomUint64(size_t x)
        {
            x ^= x >> 12; // a
            x ^= x << 25; // b
            x ^= x >> 27; // c
            return x * 0x2545F4914F6CDD1D;
        }

        std::vector<unsigned> nnSearch(float *query, size_t topk, size_t efrange)
        {
            unsigned currObj = 1;
            float curdist = RAND_MAX;
            PriorityQType candidate_set, topkRank;
            memset(flag.data(), 0, this->nRow+1);
            vector<unsigned> visited;
            vector<unsigned> knn;

            //find out the best seed from 32 random points
            for (size_t i = 0; i < 32; i++)
            {
                unsigned idx = randomUint64(i) % this->nRow; 

                if (flag[idx] == 1)
                {
                    continue;
                }
                
                float tmpdist = Metrics::l2dst(query, this->vectDat + idx * this->nDim, this->nDim);
                flag[idx] = 1;

                if (tmpdist < curdist)
                {
                    curdist = tmpdist;
                    currObj = idx;
                }
                visited.emplace_back(idx);
            }

            float lowerBound = curdist;
            topkRank.emplace(curdist, currObj);
            candidate_set.emplace(-curdist, currObj);

            //perform NN-Descent on the graph, starting from the selected seed
            while (!candidate_set.empty())
            {
                std::pair<float, unsigned> current_node_pair = candidate_set.top();
                candidate_set.pop();

                unsigned current_node = current_node_pair.second;
                float current_dist = -current_node_pair.first; // 注意：这里取负值是因为优先队列是最大堆

                // 遍历当前节点的所有邻居
                for (unsigned neighbor : nnGraph[current_node])
                {
                    if (flag[neighbor] == 1)
                    {
                        continue; // 如果邻居已经被访问过，跳过
                    }

                    float neighbor_dist = Metrics::l2dst(query, this->vectDat + neighbor * this->nDim, this->nDim);
                    flag[neighbor] = 1;
                    visited.emplace_back(neighbor);

                    // 如果邻居的距离小于当前的下界，更新下界并加入候选集
                    if (neighbor_dist < lowerBound)
                    {
                        lowerBound = neighbor_dist;
                    }

                    candidate_set.emplace(-neighbor_dist, neighbor); // 注意：这里取负值是因为优先队列是最大堆

                    // 如果邻居的距离小于当前最远的 topk 邻居的距离，加入 topk 排序队列
                    if (topkRank.size() < topk || neighbor_dist < topkRank.top().first)
                    {
                        topkRank.emplace(neighbor_dist, neighbor);
                        if (topkRank.size() > topk)
                        {
                            topkRank.pop();
                        }
                    }
                }
            }

            for (auto vit = visited.begin(); vit != visited.end(); vit++)
            {
                /**
                filling your code here
                **/
               flag[*vit] = 0; // 重置访问标志
            }
            visited.clear();
            int i = topkRank.size();
            knn.resize(topkRank.size());
            //collect the found nearest neigbors, ranked in ascending order
            while (!topkRank.empty() && i > 0)
            {
                i--;
                knn[i] = topkRank.top().second;
                topkRank.pop();
            }

            return knn;
        }

        ~NNSearch()
        {
            if (vectDat != nullptr)
            {
                delete[] vectDat;
                vectDat = nullptr;
            }
            this->nnGraph.clear();
        }
    };
}
