#ifndef SLAM_VO_BLUR_AWARE_TRACKER_COMPUTE_PIXEL_INTENSITY_H
#define SLAM_VO_BLUR_AWARE_TRACKER_COMPUTE_PIXEL_INTENSITY_H

#include "BA_tracker/core/common/CudaDefs.h"
#include "BA_tracker/core/common/Quaternion.h"
#include "BA_tracker/core/common/Vector.h"

namespace SLAM
{
    namespace VO
    {
        /**
         *  This function interpolates the intensity of pixel P2d from I_ref,
         *  and returns the interpolated intensity as well as the gradients.
         * 
         *  @param I the pointer to a 1 channel image data;
         *  @param dIx the pointer to the gradient image of I in x direction;
         *  @param dIy the pointer to the gradient image of I in y direction;
         *  @param im_H the height of the image;
         *  @param im_W the width of the image;
         *  @param P2d the pixel position;
         *  @param I_and_dI the interpolated intensity and gradients;
         *  @return the pixel is successfully interpolated;
         */
        template <typename T>
        __CPU_AND_CUDA_CODE__ bool
        bilinear_interpolation(const unsigned char *I,
                               const float *dIxy,
                               const int im_H,
                               const int im_W,
                               const Core::VectorX<T, 2> &P2d,
                               Core::Vector3d &I_and_dI)
        {
            // check if the pixel is inside the image
            if (P2d.values[0] < 0 || P2d.values[0] > im_W - 1 || P2d.values[1] < 0 || P2d.values[1] > im_H - 1)
            {
                return false;
            }

            const int xi = P2d.values[0];
            const int yi = P2d.values[1];

            float dx = P2d.values[0] - xi;
            float dy = P2d.values[1] - yi;
            float dxdy = dx * dy;

            float w00 = 1.0f - dx - dy + dxdy;
            float w01 = dx - dxdy;
            float w10 = dy - dxdy;
            float w11 = dxdy;

            int index = yi * im_W + xi;
            I_and_dI(0) = w11 * I[index + im_W + 1] +
                          w10 * I[index + im_W] +
                          w01 * I[index + 1] +
                          w00 * I[index];

            if (dIxy != nullptr)
            {
                I_and_dI(1) = w11 * dIxy[2 * (index + im_W + 1)] +
                              w10 * dIxy[2 * (index + im_W)] +
                              w01 * dIxy[2 * (index + 1)] +
                              w00 * dIxy[2 * index];

                I_and_dI(2) = w11 * dIxy[2 * (index + im_W + 1) + 1] +
                              w10 * dIxy[2 * (index + im_W) + 1] +
                              w01 * dIxy[2 * (index + 1) + 1] +
                              w00 * dIxy[2 * index + 1];
            }

            return true;
        }

        /**
         *  This function warps a pixel from current view to reference view and get the 
         *  corresponding interpolated pixel intensity.
         *  
         *  @param I_ref the pointer to a 1 channel reference image data;
         *  @param dIxy_ref the pointer to the gradient image of I_ref in xy direction;
         *  @param I_H the height of I_ref in pixels;
         *  @param I_W the width of I_ref in pixels;
         *  @param R_c2r the pointer to the unit quaternion, which is from current view to reference view;
         *  @param t_c2r the pointer to the translation vector, which is from current view to reference view;
         *  @param plane_depth the depth of the frontal parallel plane in reference view;
         *  @param intrinsics the intrinsic parameters of a pinhole camera model, i.e., fx, fy, cx, cy;
         *  @param cur_xy the pixel location in current view that we want to get its interpolated intensity;
         *  @param intensity the interpolated pixel intensity;
         *  @param jacobian the jacobian (1x7) of the interpolated pixel intensity with respect to R_c2r & t_c2r;
         *  @return the pixel intensity is successfully interpolated;
         */
        template <typename T>
        __CPU_AND_CUDA_CODE__ bool
        compute_pixel_intensity(const unsigned char *I_ref,
                                const float *dIxy_ref,
                                const int I_H,
                                const int I_W,
                                const T *R_c2r,
                                const T *t_c2r,
                                const T plane_depth,
                                const T fx,
                                const T fy,
                                const T cx,
                                const T cy,
                                const Core::VectorX<T, 2> &cur_xy,
                                T *intensity,
                                T *jacobian = nullptr)
        {
            const T qx = R_c2r[0];
            const T qy = R_c2r[1];
            const T qz = R_c2r[2];
            const T qw = R_c2r[3];
            const T x = t_c2r[0];
            const T y = t_c2r[1];
            const T z = t_c2r[2];

            // compute uray
            T x_hat = (cur_xy.values[0] - cx) / fx;
            T y_hat = (cur_xy.values[1] - cy) / fy;
            const T z_hat = 1. / sqrtf(1. + x_hat * x_hat + y_hat * y_hat);
            x_hat *= z_hat;
            y_hat *= z_hat;

            // compute lambda
            const T lambda = 2. * x_hat * (qx * qz - qw * qy) +
                             2. * y_hat * (qx * qw + qy * qz) +
                             z_hat * (qw * qw - qx * qx - qy * qy + qz * qz);

            const T DminusZoverLamda = (plane_depth - z) / lambda;

            const Core::Vector3d P3dc(DminusZoverLamda * x_hat,
                                      DminusZoverLamda * y_hat,
                                      DminusZoverLamda * z_hat);

            const Core::Vector3d P3d = Core::Quaterniond(qx, qy, qz, qw) * P3dc + Core::Vector3d(x, y, z);

            // project to image plane
            const T iz = 1.f / (P3d.values[2] + 1e-8);
            const T p2d_x = P3d.values[0] * iz;
            const T p2d_y = P3d.values[1] * iz;

            // normalize to image coordinate frame
            Core::VectorX<T, 2> ref_xy;
            ref_xy.values[0] = fx * p2d_x + cx;
            ref_xy.values[1] = fy * p2d_y + cy;

            // get interpolated pixel intensity
            Core::Vector3d I_dI(0, 0, 0);
            if (!bilinear_interpolation<T>(I_ref, dIxy_ref, I_H, I_W, ref_xy, I_dI))
            {
                return false;
            }
            *intensity = I_dI(0);

            // compute jacobian to x,y,z,qx,qy,qz,qw from Intensity
            if (jacobian != nullptr)
            {
                //
                const T T0 = qx * x_hat + qy * y_hat + qz * z_hat;
                const T T1 = qy * x_hat - qw * z_hat - qx * y_hat;
                const T T2 = qw * y_hat - qx * z_hat + qz * x_hat;
                const T T3 = qw * x_hat + qy * z_hat - qz * y_hat;
                const T T4 = qw * z_hat + qx * y_hat - qy * x_hat;

                const T C1 = 1. / (2.f * (x_hat * (-qw * qy + qx * qz) + y_hat * (qw * qx + qy * qz)) +
                                   z_hat * (qw * qw - qx * qx - qy * qy + qz * qz));

                const T K = -2.f * (qw * qz - qx * qy);
                const T L = 2.f * (qw * qy + qx * qz);
                const T M = 2.f * (qw * qz + qx * qy);
                const T N = -2.f * (qw * qx - qy * qz);
                const T P = -2.f * (qw * qy - qx * qz);
                const T Q = 2.f * (qw * qx + qy * qz);

                const T R0 = x_hat * (qw * qw + qx * qx - qy * qy - qz * qz) + y_hat * K + z_hat * L;
                const T R1 = x_hat * M + y_hat * (qw * qw - qx * qx + qy * qy - qz * qz) + z_hat * N;
                const T R2 = x_hat * P + y_hat * Q + z_hat * (qw * qw - qx * qx - qy * qy + qz * qz);

                //
                const T dPx_dqx = 2.f * (plane_depth - z) * C1 * (T0 - T2 * C1 * R0);
                const T dPy_dqx = 2.f * (plane_depth - z) * C1 * (T1 - T2 * C1 * R1);
                const T dPz_dqx = 2.f * (plane_depth - z) * C1 * (T2 - T2 * C1 * R2);

                const T dPx_dqy = 2.f * (plane_depth - z) * C1 * (T4 + T3 * C1 * R0);
                const T dPy_dqy = 2.f * (plane_depth - z) * C1 * (T0 + T3 * C1 * R1);
                const T dPz_dqy = 2.f * (plane_depth - z) * C1 * (T3 * C1 * R2 - T3);

                const T dPx_dqz = -2.f * (plane_depth - z) * C1 * (T2 + T0 * C1 * R0);
                const T dPy_dqz = 2.f * (plane_depth - z) * C1 * (T3 - T0 * C1 * R1);
                const T dPz_dqz = 2.f * (plane_depth - z) * C1 * (T0 - T0 * C1 * R2);

                const T dPx_dqw = 2.f * (plane_depth - z) * C1 * (T3 - T4 * C1 * R0);
                const T dPy_dqw = 2.f * (plane_depth - z) * C1 * (T2 - T4 * C1 * R1);
                const T dPz_dqw = 2.f * (plane_depth - z) * C1 * (T4 - T4 * C1 * R2);

                // projection jacobian
                const T dI_dPx = I_dI(1) * iz * fx;
                const T dI_dPy = I_dI(2) * iz * fy;
                const T dI_dPz = -iz * iz * (I_dI(1) * P3d.values[0] * fx + I_dI(2) * P3d.values[1] * fy);

                jacobian[0] = dI_dPx;
                jacobian[1] = dI_dPy;
                jacobian[2] = dI_dPz * (1.f - R2 * C1) - dI_dPx * R0 * C1 - dI_dPy * R1 * C1;
                jacobian[3] = dI_dPx * dPx_dqx + dI_dPy * dPy_dqx + dI_dPz * dPz_dqx;
                jacobian[4] = dI_dPx * dPx_dqy + dI_dPy * dPy_dqy + dI_dPz * dPz_dqy;
                jacobian[5] = dI_dPx * dPx_dqz + dI_dPy * dPy_dqz + dI_dPz * dPz_dqz;
                jacobian[6] = dI_dPx * dPx_dqw + dI_dPy * dPy_dqw + dI_dPz * dPz_dqw;
            }
            return true;
        }
    } // namespace VO
} // namespace SLAM

#endif
