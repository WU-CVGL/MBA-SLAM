#include <Eigen/Dense>
#include <cuda_runtime_api.h>

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
                                         double *ctrl_knot_H_cpu,
                                         double *ctrl_knot_g_cpu)
        {
            const int ndim = spline_deg_k * 6 + 1;
            const int num_elems = (ndim + 1) * ndim / 2;
            double *frame_cost_gradient_hessian_cpu = new double[num_frames * num_elems];
            cudaMemcpy(frame_cost_gradient_hessian_cpu,
                       frame_cost_gradient_hessian_gpu,
                       sizeof(double) * num_frames * num_elems,
                       cudaMemcpyDeviceToHost);

            *total_cost = 0;

            const int ctrl_knot_H_cpu_W = num_ctrl_knots * 6;

            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
                H(ctrl_knot_H_cpu, ctrl_knot_H_cpu_W, ctrl_knot_H_cpu_W);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> g(ctrl_knot_g_cpu, ctrl_knot_H_cpu_W, 1);

            if (ctrl_knot_H_cpu != nullptr)
            {
                H.setZero();
                g.setZero();
            }

            for (int i = 0; i < num_frames; i++)
            {
                const int ctrl_knot_start_idx = ctrl_knot_start_indices[i];
                const double *frame_i_cost_gradient_hessian = frame_cost_gradient_hessian_cpu + i * num_elems;

                *total_cost += frame_i_cost_gradient_hessian[0];

                if (ctrl_knot_H_cpu == nullptr)
                {
                    continue;
                }

                // gradient
                int shift = ctrl_knot_start_idx * 3;
                for (int j = 0; j < 3 * spline_deg_k; ++j)
                {
                    g(shift++) += frame_i_cost_gradient_hessian[j + 1];
                }

                shift = (num_ctrl_knots + ctrl_knot_start_idx) * 3;
                for (int j = 3 * spline_deg_k; j < 6 * spline_deg_k; ++j)
                {
                    g(shift++) += frame_i_cost_gradient_hessian[j + 1];
                }

                // hessian
                const int offset0 = ctrl_knot_start_idx * 3;
                const int offset1 = (num_ctrl_knots + ctrl_knot_start_idx) * 3;
                const double *data_ptr = frame_i_cost_gradient_hessian + ndim;

                for (int j = 0; j < ndim - 1; ++j)
                {
                    const int offset_r = j < spline_deg_k * 3 ? offset0 : offset1 - spline_deg_k * 3;
                    const int r = j + offset_r;
                    for (int k = j; k < ndim - 1; ++k, ++data_ptr)
                    {
                        const int offset_c = k < spline_deg_k * 3 ? offset0 : offset1 - spline_deg_k * 3;
                        const int c = k + offset_c;
                        const double value = *data_ptr;
                        H(r, c) += value;
                        if (c == r)
                        {
                            continue;
                        }
                        H(c, r) += value;
                    }
                }
            }
        }
    } // namespace VO
} // namespace SLAM
