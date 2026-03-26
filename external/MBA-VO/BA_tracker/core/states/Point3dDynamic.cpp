#include "Point3dDynamic.h"

namespace SLAM
{
    namespace Core
    {
        Point3dDynamic::Point3dDynamic(int ptIdx,
                                       int frameIdx,
                                       Eigen::Vector3d P3d_XYZ,
                                       MotionStatus &motion_status,
                                       double creation_time)
            : Point3dBase(ptIdx, frameIdx, P3d_XYZ),
              mSceneFlow(0, 0, 0),
              mMutableSceneFlow(0, 0, 0),
              mMotionStatus(motion_status),
              mMotionStatusUpdateTime(creation_time)
        {
            mMutableP3d_KfSceneFlows.push_back(P3d_XYZ);
        }

        Eigen::Vector3d Point3dDynamic::getSceneFlow()
        {
            std::lock_guard<std::mutex> lock(mMutexUpdate);
            return mSceneFlow;
        }

        double *Point3dDynamic::getMutableP3dData()
        {
            return mMutableP3d_KfSceneFlows.at(0).data();
        }

        double *Point3dDynamic::getMutableSceneFlow()
        {
            return mMutableSceneFlow.data();
        }

        void Point3dDynamic::invalidate()
        {
            std::lock_guard<std::mutex> lock(mMutexUpdate);
            mP3d = mMutableP3d_KfSceneFlows.at(0);
            mSceneFlow = mMutableSceneFlow;
            if (mMutableP3d_KfSceneFlows.size() > 1)
            {
                memcpy(mKfSceneFlows.data(), mMutableP3d_KfSceneFlows.at(1).data(), sizeof(Eigen::Vector3d) * mKfSceneFlows.size());
            }
        }

        void Point3dDynamic::addPointObservation(Observation &obs)
        {
            mObservations.push_back(obs);
            if (obs.frameIdx != mFrameIdx)
            {
                mKfSceneFlows.push_back(obs.scene_flow);
                mMutableP3d_KfSceneFlows.push_back(obs.scene_flow);
            }
        }

        std::vector<Point3dDynamic::Observation> &Point3dDynamic::getPointObservations()
        {
            return mObservations;
        }

        int Point3dDynamic::getNumberOfObservations()
        {
            return mObservations.size();
        }

        void Point3dDynamic::removeObservation(int frameIdx)
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

        double Point3dDynamic::getAge(double current_time)
        {
            return current_time - mMotionStatusUpdateTime;
        }

        void Point3dDynamic::updateMotionStatus(MotionStatus &status, double current_time)
        {
            mMotionStatus = status;
            mMotionStatusUpdateTime = current_time;
        }

        MotionStatus &Point3dDynamic::getMotionStatus()
        {
            return mMotionStatus;
        }
    } // namespace Core
} // namespace SLAM