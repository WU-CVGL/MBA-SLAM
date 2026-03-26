#include "Point3dBase.h"

namespace SLAM
{
    namespace Core
    {
        Point3dBase::Point3dBase(int ptIdx, int frameIdx, Eigen::Vector3d P3d_XYZ)
            : mPtIdx(ptIdx),
              mFrameIdx(frameIdx),
              mP3d(P3d_XYZ)
        {
        }

        int Point3dBase::getIdx()
        {
            return mPtIdx;
        }

        int Point3dBase::getFrameIdx()
        {
            return mFrameIdx;
        }

        Eigen::Vector3d Point3dBase::getP3d()
        {
            return mP3d;
        }
    } // namespace Core
} // namespace SLAM