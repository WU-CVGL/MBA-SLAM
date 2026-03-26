#ifndef __SRC_CORE_IMAGE_H_
#define __SRC_CORE_IMAGE_H_

#include <cstring>
#include <cstdio>
#include <memory>

#include "BA_tracker/core/common/CudaDefs.h"


namespace SLAM
{
    namespace Core
    {
        template <typename T>
        class Image
        {
        public:
            using Ptr = std::shared_ptr<Image<T>>;
            // Suppress the default copy constructor and assignment operator
            Image(const Image &);
            Image &operator=(const Image &);

            // define constructors
            Image();
            Image(size_t H, size_t W, size_t C);
            ~Image() { this->free(); }

        public:
            T *getData();
            T *getGpuData();
            T *getData(size_t r, size_t c);
            size_t nHeight();
            size_t nWidth();
            size_t nChannels();

        public:
            void copyFrom(T *dataptr, size_t H, size_t W, size_t C);
            void free();

            void uploadToGpu();

        private:
            void allocate(size_t H, size_t W, size_t C);

        private:
            T *m_data_cpu;
            T *m_data_gpu;

            size_t m_nHeight;
            size_t m_nWidth;
            size_t m_nChannels;
        };

        /* ===================================================================================*
         *                            Class implementations                                   *
         * ===================================================================================*/
        template <class T>
        Image<T>::Image()
        {
            m_data_cpu = nullptr;
            m_nHeight = 0;
            m_nWidth = 0;
            m_nChannels = 0;
            m_data_gpu = nullptr;
        }

        template <class T>
        Image<T>::Image(size_t H, size_t W, size_t C)
        {
            this->allocate(H, W, C);
        }

        template <class T>
        T *Image<T>::getData() { return m_data_cpu; }

        template <class T>
        T *Image<T>::getGpuData()
        {
            return m_data_gpu;
        }

        template <class T>
        T *Image<T>::getData(size_t r, size_t c)
        {
            return m_data_cpu + r * m_nWidth * m_nChannels + c * m_nChannels;
        }

        template <class T>
        size_t Image<T>::nHeight() { return m_nHeight; }

        template <class T>
        size_t Image<T>::nWidth() { return m_nWidth; }

        template <class T>
        size_t Image<T>::nChannels() { return m_nChannels; }

        template <class T>
        void Image<T>::copyFrom(T *dataptr, size_t H, size_t W, size_t C)
        {
            bool reallocate = !(H == m_nHeight && W == m_nWidth && C == m_nChannels);

            if (reallocate)
            {
                this->free();

                m_nHeight = H;
                m_nWidth = W;
                m_nChannels = C;

                this->allocate(H, W, C);
            }

            std::memcpy(this->m_data_cpu, dataptr, H * W * C * sizeof(T));
        }

        template <class T>
        void Image<T>::free()
        {
            delete[] m_data_cpu;
            m_data_cpu = nullptr;
            m_nHeight = 0;
            m_nWidth = 0;
            m_nChannels = 0;

#ifdef COMPILE_WITH_CUDA
            cudaFree(m_data_gpu);
#endif
        }

        template <class T>
        void Image<T>::uploadToGpu()
        {
            if (m_data_cpu == nullptr || m_data_gpu != nullptr)
            {
                printf("%s: m_data_cpu=%d; m_data_gpu=%d\n", __FUNCTION__, m_data_cpu, m_data_gpu);
                return;
            }
#ifdef COMPILE_WITH_CUDA
            // printf("Allocating GPU memory\n");
            cudaMalloc((void **)&m_data_gpu, sizeof(T) * m_nHeight * m_nWidth * m_nChannels);
            cudaMemcpy(m_data_gpu, m_data_cpu, sizeof(T) * m_nHeight * m_nWidth * m_nChannels, cudaMemcpyHostToDevice);
#endif
        }

        template <class T>
        void Image<T>::allocate(size_t H, size_t W, size_t C)
        {
            m_data_cpu = new T[H * W * C];
            m_nHeight = H;
            m_nWidth = W;
            m_nChannels = C;
            m_data_gpu = nullptr;
        }
    } /* namespace Core */
} /* namespace SLAM */

#endif
