//
// Created by peidong on 2/14/20.
//

#ifndef SLAM_INPUTOUTPUT_H
#define SLAM_INPUTOUTPUT_H

#include "BA_tracker/core/measurements/Image.h"
#include "BA_tracker/core/measurements/ImuMeasurement.h"
#include "BA_tracker/core/states/NavState.h"
#include <string>
#include <vector>

namespace SLAM
{
    namespace Utils {
        void load_depthMap(std::string path_to_file, SLAM::Core::Image<float>::Ptr depthMap);

        void save_pcl_to_plyFile(Core::Image<float> *pcl, std::string save_file_path);

        bool unreal_get_ground_truth_nav_state(std::string    path_to_pose,
                                              double          _t,
                                              Core::NavState *navState);

        bool unreal_get_imu_measurements(std::string path_to_pose, 
                                        double t_start, 
                                        double t_end, 
                                        std::vector<Core::ImuMeasurement> &imuMeasurements);
        

    }
}

#endif //SLAM_UTILSFUNCTIONS_H
