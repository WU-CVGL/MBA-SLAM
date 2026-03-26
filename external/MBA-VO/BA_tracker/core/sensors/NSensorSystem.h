//
// Created by peidong on 2/13/20.
//

#ifndef SLAM_NSENSORSYSTEM_H
#define SLAM_NSENSORSYSTEM_H

#include "CameraBase.h"
#include "Imu.h"
#include <map>
#include <string>
#include <vector>

namespace SLAM
{
    namespace Core
    {
        class NSensorSystem
        {
        public:
            NSensorSystem();
            ~NSensorSystem();

        public:
            void add_camera(int devId, CameraBase *cameraPtr);
            void add_camera(int devId, std::string &devName, CameraBase *cameraPtr);
            void add_paired_camera(int refCamDevId, int overlappedCamDevId);

            void add_imu(int devId, IMU *imuPtr);

            CameraBase *get_camera(int devId);
            const std::map<int, CameraBase *> &get_cameras() const;
            const std::map<int, int> &get_paired_cameras() const;

            IMU *get_imu(int devId);

            int get_devId(const std::string &devName);

        private:
            // has ownership to all sensors
            std::map<int, CameraBase *> m_NCamDevNamePtr;
            // <refCameraId, overlappedCameraId>
            std::map<int, int> m_RefCamOverlappedCamDevId;

            std::map<int, IMU *> m_NImuDevPtr;

            // <camDevName, camId>
            std::map<std::string, int> m_NCamDevNameIdMap;
        };
    } // namespace Core
} // namespace SLAM

#endif