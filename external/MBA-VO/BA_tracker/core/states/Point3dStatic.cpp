#include "Point3dStatic.h"

namespace SLAM
{
    namespace Core
    {
        Point3dStatic::Point3dStatic(int ptIdx, int frameIdx, Eigen::Vector3d P3d_XYZ)
            : Point3dBase(ptIdx, frameIdx, P3d_XYZ),
              mMutableP3d(P3d_XYZ)
        {
        }

        void Point3dStatic::invalidate()
        {
            std::lock_guard<std::mutex> lock(mMutexUpdate);
            mMutableP3d = mP3d;
        }

        int Point3dStatic::getNumberOfObservations()
        {
            return mObservations.size();
        }

        double *Point3dStatic::getMutableP3dData()
        {
            return mMutableP3d.data();
        }

        void Point3dStatic::addPointObservation(Observation &obs)
        {
            mObservations.push_back(obs);
        }

        std::vector<Point3dStatic::Observation> &Point3dStatic::getPointObservations()
        {
            return mObservations;
        }

        void Point3dStatic::removeObservation(int frameIdx)
        {
            for (auto it_obs = mObservations.begin(); it_obs != mObservations.end();)
            {
                if (it_obs->frameIdx != frameIdx)
                {
                    it_obs++;
                    continue;
                }
                it_obs = mObservations.erase(it_obs);
            }
        }
    } // namespace Core
} // namespace SLAM