#include "NSensorSystem.h"

namespace SLAM
{
    namespace Core
    {
        NSensorSystem::NSensorSystem()
        {
        }

        NSensorSystem::~NSensorSystem()
        {
            for (auto it = m_NCamDevNamePtr.begin(); it != m_NCamDevNamePtr.end(); ++it)
            {
                delete it->second;
            }

            for (auto it = m_NImuDevPtr.begin(); it != m_NImuDevPtr.end(); ++it)
            {
                delete it->second;
            }
        }

        void NSensorSystem::add_camera(int devId, CameraBase *cameraPtr)
        {
            m_NCamDevNamePtr.insert(std::pair<int, CameraBase *>(devId, cameraPtr));
        }

        void NSensorSystem::add_paired_camera(int refCamDevId, int overlappedCamDevId)
        {
            m_RefCamOverlappedCamDevId.insert(std::pair<int, int>(refCamDevId, overlappedCamDevId));
        }

        void NSensorSystem::add_camera(int devId, std::string &devName, CameraBase *cameraPtr)
        {
            add_camera(devId, cameraPtr);
            m_NCamDevNameIdMap.insert(std::pair<std::string, int>(devName, devId));
        }

        void NSensorSystem::add_imu(int devId, IMU *imuPtr)
        {
            m_NImuDevPtr.insert(std::pair<int, IMU *>(devId, imuPtr));
        }

        CameraBase *NSensorSystem::get_camera(int devId)
        {
            return m_NCamDevNamePtr.at(devId);
        }

        const std::map<int, CameraBase *> &NSensorSystem::get_cameras() const
        {
            return m_NCamDevNamePtr;
        }

        const std::map<int, int> &NSensorSystem::get_paired_cameras() const
        {
            return m_RefCamOverlappedCamDevId;
        }

        IMU *NSensorSystem::get_imu(int devId)
        {
            return m_NImuDevPtr.at(devId);
        }

        int NSensorSystem::get_devId(const std::string &devName)
        {
            return m_NCamDevNameIdMap.at(devName);
        }

    } // namespace Core
} // namespace SLAM