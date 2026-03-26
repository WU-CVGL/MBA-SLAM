//
// Created by lzzhao on 12/25/23.
//

#ifndef BA_TRACKER_FAILURE_INDICATORS_H
#define BA_TRACKER_FAILURE_INDICATORS_H

namespace SLAM::VO {
struct FailureIndicators
{
    double avg_Rt_flow_mag = 0;
    double avg_blur_kernel_len = 0;
    double avg_t_flow_mag  = 0;
};
}


#endif //BA_TRACKER_FAILURE_INDICATORS_H
