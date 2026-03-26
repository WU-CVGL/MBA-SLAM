//
// Created by peidong on 2/15/20.
//

#ifndef SLAM_GRADIENT_H
#define SLAM_GRADIENT_H

#include "BA_tracker/core/measurements/Image.h"
#include <assert.h>
#include <cmath>

namespace SLAM
{
    namespace Core
    {
        template <class T1, class T2>
        void compute_image_gradients(
                typename Image<T1>::Ptr srcImg,
                typename Image<T2>::Ptr gradImg,
                Image<float>::Ptr magImg = {}
        ) {
            assert(srcImg != nullptr);
            assert(srcImg->nHeight() == gradImg->nHeight());
            assert(srcImg->nWidth() == gradImg->nWidth());
            assert(srcImg->nChannels() * 2 == gradImg->nChannels());

            size_t H = srcImg->nHeight();
            size_t W = srcImg->nWidth();
            size_t C = srcImg->nChannels();
            T1 *srcImgPtr = srcImg->getData();
            T2 *gradImgPtr = gradImg->getData();
            float *magImgPtr = nullptr;
            if (magImg != nullptr)
                magImgPtr = magImg->getData();

            for (int y = 0; y < H; y++)
            {
                for (int x = 0; x < W; ++x, srcImgPtr += C, gradImgPtr += 2 * C)
                {
                    if (x == 0 || y == 0 || x == W - 1 || y == H - 1)
                    {
                        for (int c = 0; c < C; ++c)
                        {
                            gradImgPtr[2 * c] = 0;
                            gradImgPtr[2 * c + 1] = 0;
                        }
                        if (magImgPtr != nullptr)
                        {
                            magImgPtr[0] = 0;
                            magImgPtr++;
                        }
                        continue;
                    }

                    T1 *lptr = srcImgPtr - C;
                    T1 *rptr = srcImgPtr + C;
                    T1 *tptr = srcImgPtr - W * C;
                    T1 *bptr = srcImgPtr + W * C;

                    float mag = 0;
                    for (int c = 0; c < C; ++c)
                    {
                        T2 dx = 0.5 * ((T2)rptr[c] - (T2)lptr[c]);
                        T2 dy = 0.5 * ((T2)bptr[c] - (T2)tptr[c]);

                        gradImgPtr[2 * c] = dx;
                        gradImgPtr[2 * c + 1] = dy;

                        mag += sqrt(dx * dx + dy * dy);
                    }
                    if (magImgPtr != nullptr)
                    {
                        magImgPtr[0] = mag / C;
                        magImgPtr++;
                    }
                }
            }
        }

        template <class T1, class T2, class T3>
        void compute_image_gradients(
                typename Image<T1>::Ptr srcImg,
                typename Image<T2>::Ptr gradImgX,
                typename Image<T3>::Ptr gradImgY,
                typename Image<float>::Ptr magImg = nullptr
        ) {
            assert(srcImg != nullptr);
            assert(srcImg->nHeight() == gradImgX->nHeight());
            assert(srcImg->nWidth() == gradImgX->nWidth());
            assert(srcImg->nChannels() == gradImgX->nChannels());
            assert(srcImg->nChannels() == gradImgY->nChannels());

            size_t H = srcImg->nHeight();
            size_t W = srcImg->nWidth();
            size_t C = srcImg->nChannels();
            T1 *srcImgPtr = srcImg->getData();
            T2 *gradImgXPtr = gradImgX->getData();
            T2 *gradImgYPtr = gradImgY->getData();
            float *magImgPtr = nullptr;
            if (magImg != nullptr)
            {
                magImgPtr = magImg->getData();
            }

            for (int y = 0; y < H; y++)
            {
                for (int x = 0; x < W; ++x, srcImgPtr += C, gradImgXPtr += C, gradImgYPtr += C)
                {
                    if (x == 0 || y == 0 || x == W - 1 || y == H - 1)
                    {
                        for (int c = 0; c < C; ++c)
                        {
                            gradImgXPtr[c] = 0;
                            gradImgYPtr[c] = 0;
                        }
                        if (magImgPtr != nullptr)
                        {
                            magImgPtr[0] = 0;
                            magImgPtr++;
                        }
                        continue;
                    }

                    T1 *lptr = srcImgPtr - C;
                    T1 *rptr = srcImgPtr + C;
                    T1 *tptr = srcImgPtr - W * C;
                    T1 *bptr = srcImgPtr + W * C;

                    float mag = 0;
                    for (int c = 0; c < C; ++c)
                    {
                        T2 dx = 0.5 * ((T2)rptr[c] - (T2)lptr[c]);
                        T2 dy = 0.5 * ((T2)bptr[c] - (T2)tptr[c]);

                        gradImgXPtr[c] = dx;
                        gradImgYPtr[c] = dy;

                        mag += sqrt(dx * dx + dy * dy);
                    }
                    if (magImgPtr != nullptr)
                    {
                        magImgPtr[0] = mag / C;
                        magImgPtr++;
                    }
                }
            }
        }
    } // namespace Core
} // namespace SLAM

#endif //SLAM_GRADIENT_H
