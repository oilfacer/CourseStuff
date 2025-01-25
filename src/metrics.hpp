#pragma once;

#include <stdlib.h>


/***
 * @author Wan-Lei Zhao
 * @date   2024-12-10
 * 
 * @copyright All rights are reserved by the author
 */

namespace cmmlab
{

    class Metrics
    {
    public:
        static float l2dst(float *vect1, float *vect2, size_t dim)
        {
            float dist = 0, delta = 0;
            for (unsigned i = 0; i < dim; i++)
            {
                delta = vect1[i] - vect2[i];
                dist += delta * delta;
            }
            return dist;
        }
    };
}