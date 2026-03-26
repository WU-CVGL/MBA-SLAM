#include "MapSparse.h"
#include <limits>

namespace SLAM
{
    namespace Core
    {
        MapSparse::MapSparse()
        {
            mUniquePointIdx = 0;
        }

        MapSparse::~MapSparse()
        {
            for (int i = 0; i < mPointCloud.size(); ++i)
            {
                delete mPointCloud.at(i);
            }
        }

        void MapSparse::addPoint(Point3dBase *P3d)
        {
            mPointCloud.insert(std::pair<int, Point3dBase *>(P3d->getIdx(), P3d));
        }

        Point3dBase *MapSparse::getPoint(int pointIdx)
        {
            return mPointCloud.at(pointIdx);
        }

        std::map<int, Point3dBase *> &MapSparse::getPointCloud()
        {
            return mPointCloud;
        }

        size_t MapSparse::getUniquePointIdx()
        {
            mUniquePointIdx++;
            mUniquePointIdx %= std::numeric_limits<size_t>::max();
            return mUniquePointIdx;
        }

        size_t MapSparse::getNumberOfObservations()
        {
            size_t N = 0;
            for (auto it = mPointCloud.begin(); it != mPointCloud.end(); it++)
            {
                N += it->second->getNumberOfObservations();
            }
            return N;
        }
    } // namespace Core
} // namespace SLAM
