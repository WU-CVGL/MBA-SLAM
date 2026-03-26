//
// Created by peidong on 4/10/20.
//

#ifndef SLAM_MAP_H
#define SLAM_MAP_H

#include "Point3dBase.h"
#include <map>
#include <vector>

namespace SLAM
{
    namespace Core
    {
        class MapSparse
        {
        public:
            MapSparse();
            ~MapSparse();

        public:
            void addPoint(Point3dBase *P3d);

            Point3dBase *getPoint(int pointIdx);
            std::map<int, Point3dBase *> &getPointCloud();

            size_t getUniquePointIdx();
            size_t getNumberOfObservations();

        private:
            // <pointIdx, Point3d*>
            std::map<int, Point3dBase *> mPointCloud; // have ownership

            size_t mUniquePointIdx;
        };
    } // namespace Core
} // namespace SLAM

#endif
