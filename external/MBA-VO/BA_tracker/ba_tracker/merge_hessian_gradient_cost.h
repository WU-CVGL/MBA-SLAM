#ifndef SLAM_VO_MERGE_HESSIAN_GRADIENT_COST_H
#define SLAM_VO_MERGE_HESSIAN_GRADIENT_COST_H

namespace SLAM
{
    namespace VO
    {
        void merge_hessian_gradient_cost(const int num_frames,
                                         const int spline_deg_k,
                                         const double *frame_cost_gradient_hessian_gpu,
                                         const int *ctrl_knot_start_indices,
                                         const int num_ctrl_knots,
                                         double *total_cost,
                                         double *ctrl_knot_H_cpu = nullptr,
                                         double *ctrl_knot_g_cpu = nullptr);
    } // namespace VO
} // namespace SLAM

#endif