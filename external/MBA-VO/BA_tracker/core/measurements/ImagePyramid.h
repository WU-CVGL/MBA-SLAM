//
// Created by peidong on 2/16/20.
//

#ifndef SLAM_IMAGEPYRAMID_H
#define SLAM_IMAGEPYRAMID_H

#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include "Image.h"


namespace SLAM
{
    namespace Core
    {
        template<class T>
        class ImagePyramid {
        public:
            using Ptr = std::shared_ptr<ImagePyramid>;
            ImagePyramid();
            ~ImagePyramid();

        public:
            void setNumOfPyramidLevels(int nLevels);
            void setPyramid(std::vector<typename Image<T>::Ptr> imagePyramidPtrList);
            void computePyramid(typename Image<T>::Ptr imagePtr);
            void copyPyramidFrom(int level, T* dataptr, int H, int W, int C);

        public:
            typename Image<T>::Ptr getImagePtr(int level);
            int getNumOfPyramidLevels();

        private:
            int m_nLevels;
            std::map<int, typename Image<T>::Ptr> m_imagePyramidPtrMap; // (level, image): has ownership
        };

        /* ===================================================================================*
         *                            Class implementations                                   *
         * ===================================================================================*/
        template<class T>
        ImagePyramid<T>::ImagePyramid()
        {
            m_nLevels = 0;
        }

        template<class T>
        ImagePyramid<T>::~ImagePyramid()
        {
        }

        template<class T>
        void ImagePyramid<T>::setNumOfPyramidLevels(int nLevels)
        {
            m_nLevels = nLevels;
        }

        template<class T>
        void ImagePyramid<T>::computePyramid(typename Image<T>::Ptr imagePtr)
        {
            int H0 = imagePtr->nHeight();
            int W0 = imagePtr->nWidth();
            int C = imagePtr->nChannels();
            this->copyPyramidFrom(0, imagePtr->getData(), H0, W0, C);

            for(int lv = 1; lv < m_nLevels; lv++)
            {
                T* dataPtr = m_imagePyramidPtrMap.at(lv-1)->getData();
                int Wlv_1 = m_imagePyramidPtrMap.at(lv-1)->nWidth();

                int Hlv = H0 / int(pow(2, lv));
                int Wlv = W0 / int(pow(2, lv));
                auto image_lv = std::make_shared<Image<T>>(Hlv, Wlv, C);
                T* dataPtr_lv = image_lv->getData();

                // compute image
                for(int h = 0; h < Hlv; ++h)
                {
                    for(int w = 0; w < Wlv; ++w)
                    {
                        T* dataPtr00 = dataPtr + 2*h*Wlv_1*C +2*w*C;
                        T* dataPtr01 = dataPtr + 2*h*Wlv_1*C +(2*w+1)*C;
                        T* dataPtr10 = dataPtr + (2*h+1)*Wlv_1*C +2*w*C;
                        T* dataPtr11 = dataPtr + (2*h+1)*Wlv_1*C +(2*w+1)*C;

                        for(int c=0; c < C; ++c, ++dataPtr_lv)
                        {
                            dataPtr_lv[0] = T(0.25*(float(dataPtr00[c])+
                                                    float(dataPtr01[c])+
                                                    float(dataPtr10[c])+
                                                    float(dataPtr11[c])));
                        }
                    }
                }

                //
                m_imagePyramidPtrMap.insert(std::pair<int, typename Image<T>::Ptr>(lv, image_lv));
            }
        }

        template<class T>
        void ImagePyramid<T>::setPyramid(std::vector<typename Image<T>::Ptr> imagePyramidPtrList)
        {
            m_imagePyramidPtrMap.clear();
            int lv = 0;
            for(auto imagePtr : imagePyramidPtrList)
            {
                m_imagePyramidPtrMap.insert(std::pair<int, typename Image<T>::Ptr>(lv, imagePtr));
                lv++;
            }
            m_nLevels = lv;
        }

        template<class T>
        void ImagePyramid<T>::copyPyramidFrom(int level, T *dataptr, int H, int W, int C)
        {
            auto image = std::make_shared<Image<T>>(H, W, C);
            image->copyFrom(dataptr, H, W, C);
            m_imagePyramidPtrMap.insert(std::pair<int, typename Image<T>::Ptr> (level, image));
        }

        template<class T>
        typename Image<T>::Ptr ImagePyramid<T>::getImagePtr(int level)
        {
            return m_imagePyramidPtrMap.at(level);
        }

        template<class T>
        int ImagePyramid<T>::getNumOfPyramidLevels()
        {
            return m_nLevels;
        }
    }
}

#endif //SLAM_IMAGEPYRAMID_H
