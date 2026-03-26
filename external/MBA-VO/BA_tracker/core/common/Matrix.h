#ifndef SLAM_CORE_MATRIX_H_
#define SLAM_CORE_MATRIX_H_

#include "BA_tracker/core/common/CudaDefs.h"

namespace SLAM
{
    namespace Core
    {
        template <class T, int nRows_, int nCols_>
        struct MatrixXX_
        {
            int nRows;
            int nCols;
            T values[nRows_ * nCols_];
        };

        template <class T, int nRows_, int nCols_>
        class MatrixXX : public MatrixXX_<T, nRows_, nCols_>
        {
        public:
            __CPU_AND_CUDA_CODE__ MatrixXX()
            {
                this->nRows = nRows_;
                this->nCols = nCols_;
            }

            __CPU_AND_CUDA_CODE__ T &operator()(int i, int j)
            {
                return *(this->values + i * this->nCols + j);
            }
        };
    } // namespace Core
} // namespace SLAM

#endif