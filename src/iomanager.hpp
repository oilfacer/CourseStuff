#pragma once

#include <iostream>
#include <stdlib.h>
#include <malloc.h>
#include <iterator>
#include <fstream>
#include <cstring>
#include <sstream>
#include <assert.h>
#include <vector>
#include <string>

using namespace std;

/**
 * @author: Jie-Feng Wang, Wan-Lei Zhao
 * @date: 2022 - 2024
 *
 * @brief This class is responsible for the input/output operations
 * for this project.
 * 1. load raw vectors from a specified file
 * 2. save the generated graph to the specified path
 * 3. load the graph from the specified path
 *
 * The definition and implementation of this class are slightly
 * differet from 'IOManager' in FastSearch project. Compared to that
 * class, a few more functions are defined here.
 *
 * @copyright All rights are reserved by the author
 */

namespace cmmlab
{
    class IOManager
    {

    public:
     
        static vector<vector<float>> loadFVECS(string srcPath)
        {
            vector<vector<float>> matrix;

            streampos fileSize;
            ifstream inStrm(srcPath, ios::binary);
            if (!inStrm.is_open())
            {
                std::cerr << "File '" << srcPath << "' cannot open for read!\n";
                exit(0);
            }
            unsigned int dim = 0;

            inStrm.seekg(0, ios::end);
            fileSize = inStrm.tellg();
            inStrm.seekg(0, ios::beg);
            inStrm.read((char *)&dim, sizeof(unsigned int));
            unsigned int size_n = (double)fileSize / (sizeof(unsigned int) + dim * sizeof(float));

            auto nRow = size_n;
            auto nDim = dim;

            for (size_t i = 0; i < nRow; ++i)
            {
                vector<float> tmp_row(nDim, 0.0);
                matrix.push_back(tmp_row);
            }

            inStrm.seekg(0, ios::beg);
            for (size_t i = 0; i < size_n; ++i)
            {
                inStrm.read((char *)&dim, sizeof(unsigned int));
                inStrm.read((char *)matrix[i].data(), dim * sizeof(float));
            }
            inStrm.close();
            return matrix;
        }

        static float *loadFVECSPtr(string srcPath, size_t &nRow, size_t &nDim)
        {
            streampos fileSize;
            ifstream inStrm(srcPath, ios::binary);
            if (!inStrm.is_open())
            {
                std::cerr << "File '" << srcPath << "' cannot open for read!\n";
                exit(0);
            }
            unsigned int dim = 0;
            inStrm.seekg(0, ios::end);
            fileSize = inStrm.tellg();
            inStrm.seekg(0, ios::beg);
            inStrm.read((char *)&dim, sizeof(unsigned int));
            unsigned int size_n = (double)fileSize / (sizeof(unsigned int) + dim * sizeof(float));
            std::cout << size_n << "\t" << dim << std::endl;

            nRow = size_n;
            nDim = dim;
            float *matrix = new float[nRow * nDim];
            std::cout << nRow << "\t" << nDim << std::endl;

            inStrm.seekg(0, ios::beg);
            for (size_t i = 0; i < size_n; ++i)
            {
                inStrm.read((char *)&dim, sizeof(unsigned int));
                inStrm.read((char *)(matrix + i * nDim), nDim * sizeof(float));
            }
            inStrm.close();
            return matrix;
        }

        static vector<vector<unsigned int>> loadIVECS(string srcPath, size_t &nRow, size_t &nDim)
        {
            vector<vector<unsigned int>> matrix;
            streampos fileSize;
            ifstream inStrm(srcPath, ios::binary);
            if (!inStrm.is_open())
            {
                std::cerr << "File '" << srcPath << "' cannot open for read!\n";
                exit(0);
            }
            unsigned int dim = 0;

            inStrm.seekg(0, ios::end);
            fileSize = inStrm.tellg();
            inStrm.seekg(0, ios::beg);
            inStrm.read((char *)&dim, sizeof(unsigned int));
            unsigned int size_n = (double)fileSize / (sizeof(unsigned int) + dim * sizeof(unsigned int));

            nRow = size_n;
            nDim = dim;

            for (size_t i = 0; i < nRow; ++i)
            {
                vector<unsigned int> tmp_row(nDim, 0);
                matrix.push_back(tmp_row);
            }

            inStrm.seekg(0, ios::beg);

            for (size_t i = 0; i < size_n; ++i)
            {
                inStrm.read((char *)&dim, sizeof(unsigned int));
                inStrm.read((char *)matrix[i].data(), dim * sizeof(unsigned int));
            }
            inStrm.close();
            return matrix;
        }

        static vector<vector<unsigned int>> loadIVECS(string srcPath)
        {
            ifstream inStrm(srcPath, ios::binary);
            unsigned int dim = 0;

            inStrm.seekg(0, ios::beg);
            if (!inStrm.is_open())
            {
                std::cerr << "File '" << srcPath << "' cannot open for read!\n";
                exit(0);
            }
            vector<vector<unsigned int>> matrix;

            while (inStrm.read((char *)&dim, sizeof(unsigned int)))
            {
                vector<unsigned int> row;
                row.resize(dim);
                inStrm.read((char *)row.data(), dim * sizeof(unsigned int));
                matrix.push_back(row);
                // cout << row.size() << endl;
            }
            inStrm.close();
            return matrix;
        }

         static void saveIVECS(string destPath, vector<vector<unsigned>> &data)
        {
            unsigned int size_n = data.size();
            unsigned int dim = data[0].size();

            auto outStrm = std::fstream(destPath, ios::out | ios::binary);
            if (!outStrm.is_open())
            {
                std::cerr << "File '" << destPath << "' cannot open for write!" << std::endl;
                exit(0);
            }
            for (int i = 0; i < size_n; ++i)
            {
                dim = data[i].size();
                outStrm.write((char *)&dim, sizeof(unsigned int));
                for (int j = 0; j < dim; ++j)
                {
                    outStrm.write((char *)&data[i][j], sizeof(unsigned int));
                }
            }
            outStrm.close();
        }


        static void saveNNGraph_as_TXT(string destPath, vector<vector<unsigned int>> &data)
        {
#ifdef VERBOSE
            cout << "saving " << destPath << endl;
#endif
            ofstream output_file(destPath);
            if (!output_file.is_open())
            {
                std::cerr << "File '" << destPath << "' cannot open for write!" << std::endl;
                exit(0);
            }
            int size_n = data.size();
            int size_k = 0;
            for (int i = 0; i < size_n; ++i)
            {
                size_k = data[i].size();
                output_file << size_k;
                for (int j = 0; j < size_k; ++j)
                {
                    output_file << " " << data[i][j];
                }
                output_file << "\n";
            }
            output_file.close();
        }

        static void test()
        {
            string srcPath = "/home/wlzhao/datasets/bignn/sift1m/sift_learn.fvecs";

            vector<vector<float>> mat = IOManager::loadFVECS(srcPath);
            for(int i = 0; i < mat[0].size(); i++)
            {
                cout << mat[99990][i] << " ";
            }
        }
    };

}
