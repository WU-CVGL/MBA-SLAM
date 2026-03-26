#ifndef CORE_COMMON_QUATERNION_H_
#define CORE_COMMON_QUATERNION_H_

#include "CudaDefs.h"
#include "Vector.h"
#include <cmath>
#include <cstring>

namespace SLAM
{
    namespace Core
    {
        struct Quaterniond
        {
            double x;
            double y;
            double z;
            double w;

            __CPU_AND_CUDA_CODE__ Quaterniond()
                : x(0), y(0), z(0), w(1)
            {
            }

            __CPU_AND_CUDA_CODE__ Quaterniond(double x_, double y_, double z_, double w_)
                : x(x_), y(y_), z(z_), w(w_)
            {
            }

            __CPU_AND_CUDA_CODE__ void normalize()
            {
                double norm = sqrt(x * x + y * y + z * z + w * w);
                x /= norm;
                y /= norm;
                z /= norm;
                w /= norm;
            }

            __CPU_AND_CUDA_CODE__ Quaterniond conjugate() const
            {
                return Quaterniond(-x, -y, -z, w);
            }

            __CPU_AND_CUDA_CODE__ Quaterniond operator*(const Quaterniond &qy) const
            {
                return Quaterniond(w * qy.x + x * qy.w + y * qy.z - z * qy.y,
                                   w * qy.y + y * qy.w + z * qy.x - x * qy.z,
                                   w * qy.z + z * qy.w + x * qy.y - y * qy.x,
                                   w * qy.w - x * qy.x - y * qy.y - z * qy.z);
            }

            __CPU_AND_CUDA_CODE__ Vector3d operator*(const Vector3d &p) const
            {
                Quaterniond Qp(p(0), p(1), p(2), 0);
                Quaterniond Q(x, y, z, w);
                Quaterniond Qc(-x, -y, -z, w);
                Quaterniond V = Q * Qp * Qc;
                return Vector3d(V.x, V.y, V.z);
            }

            __CPU_AND_CUDA_CODE__ Vector3d log(double *jacobian = nullptr) const
            {
                // jacobian is a 3x4 matrix
                // ** qx qy qz qw
                // wx
                // wy
                // wz

                const double squared_n = x * x + y * y + z * z;
                double lambda;

                double dlambda_dx = 0;
                double dlambda_dy = 0;
                double dlambda_dz = 0;
                double dlambda_dw = 0;

                if (squared_n < 1e-20)
                {
                    const double www = w * w * w;
                    lambda = 2. / w - 2. / 3. * squared_n / www;

                    if (jacobian != nullptr)
                    {
                        dlambda_dx = 2. / w - 4. / 3. * x / www;
                        dlambda_dy = 2. / w - 4. / 3. * y / www;
                        dlambda_dz = 2. / w - 4. / 3. * z / www;
                        dlambda_dw = -2 / (w * w) + 2 * squared_n / (www * w);
                    }
                }
                else
                {
                    const double n = sqrt(squared_n);
                    if (fabs(w) < 1e-10)
                    {
                        if (w > 0)
                        {
                            lambda = M_PI / n;
                            if (jacobian != nullptr)
                            {
                                dlambda_dx = -lambda / squared_n * x;
                                dlambda_dy = -lambda / squared_n * y;
                                dlambda_dz = -lambda / squared_n * z;
                            }
                        }
                        else
                        {
                            lambda = -M_PI / n;
                            if (jacobian != nullptr)
                            {
                                dlambda_dx = lambda / squared_n * x;
                                dlambda_dy = lambda / squared_n * y;
                                dlambda_dz = lambda / squared_n * z;
                            }
                        }
                    }
                    else
                    {
                        lambda = 2.0 * atan(n / w) / n;
                        if (jacobian != nullptr)
                        {
                            double dlambda_dn = (2 * w - lambda) / n;
                            dlambda_dx = dlambda_dn * x / n;
                            dlambda_dy = dlambda_dn * y / n;
                            dlambda_dz = dlambda_dn * z / n;
                            dlambda_dw = -2.;
                        }
                    }
                }

                if (jacobian != nullptr)
                {
                    jacobian[0] = dlambda_dx * x + lambda;
                    jacobian[1] = dlambda_dy * x;
                    jacobian[2] = dlambda_dz * x;
                    jacobian[3] = dlambda_dw * x;

                    jacobian[4] = dlambda_dx * y;
                    jacobian[5] = dlambda_dy * y + lambda;
                    jacobian[6] = dlambda_dz * y;
                    jacobian[7] = dlambda_dw * y;

                    jacobian[8] = dlambda_dx * z;
                    jacobian[9] = dlambda_dy * z;
                    jacobian[10] = dlambda_dz * z + lambda;
                    jacobian[11] = dlambda_dw * z;
                }

                Vector3d tangent(lambda * x,
                                 lambda * y,
                                 lambda * z);
                return tangent;
            }

            __CPU_AND_CUDA_CODE__ static Quaterniond exp(Vector3d &tangent, double *jacobian = nullptr)
            {
                // jacobian is a 4x3 matrix:
                // ** wx, wy, wz
                // qx
                // qy
                // qz
                // qw

                double imag_factor, real_factor;

                const double theta_sq = tangent.squaredNorm();
                if (theta_sq < 1e-20)
                {
                    const double theta_po4 = theta_sq * theta_sq;
                    imag_factor = 0.5 - 1. / 48. * theta_sq + 1. / 3840. * theta_po4;
                    real_factor = 1. - 1. / 8. * theta_sq + 1. / 384. * theta_po4;

                    if (jacobian != nullptr)
                    {
                        std::memset(jacobian, 0, sizeof(double) * 12);
                        jacobian[0] = 0.5;
                        jacobian[4] = 0.5;
                        jacobian[8] = 0.5;
                    }
                }
                else
                {
                    const double theta = sqrt(theta_sq);
                    const double half_theta = 0.5 * theta;
                    const double sin_half_theta = sin(half_theta);

                    imag_factor = sin_half_theta / theta;
                    real_factor = cos(half_theta);

                    if (jacobian != nullptr)
                    {
                        const double x = tangent.values[0];
                        const double y = tangent.values[1];
                        const double z = tangent.values[2];

                        const double dtheta_dx = x / theta;
                        const double dtheta_dy = y / theta;
                        const double dtheta_dz = z / theta;

                        const double dimag_factor_dtheta = 0.5 * real_factor / theta - imag_factor / theta;
                        const double dreal_factor_dtheta = -0.5 * sin_half_theta;

                        //
                        const double dimag_factor_dx = dimag_factor_dtheta * dtheta_dx;
                        const double dimag_factor_dy = dimag_factor_dtheta * dtheta_dy;
                        const double dimag_factor_dz = dimag_factor_dtheta * dtheta_dz;

                        const double dreal_factor_dx = dreal_factor_dtheta * dtheta_dx;
                        const double dreal_factor_dy = dreal_factor_dtheta * dtheta_dy;
                        const double dreal_factor_dz = dreal_factor_dtheta * dtheta_dz;

                        jacobian[0] = dimag_factor_dx * x + imag_factor;
                        jacobian[1] = dimag_factor_dy * x;
                        jacobian[2] = dimag_factor_dz * x;

                        jacobian[3] = dimag_factor_dx * y;
                        jacobian[4] = dimag_factor_dy * y + imag_factor;
                        jacobian[5] = dimag_factor_dz * y;

                        jacobian[6] = dimag_factor_dx * z;
                        jacobian[7] = dimag_factor_dy * z;
                        jacobian[8] = dimag_factor_dz * z + imag_factor;

                        jacobian[9] = dreal_factor_dx;
                        jacobian[10] = dreal_factor_dy;
                        jacobian[11] = dreal_factor_dz;
                    }
                }

                return Quaterniond(imag_factor * tangent.values[0],
                                   imag_factor * tangent.values[1],
                                   imag_factor * tangent.values[2],
                                   real_factor);
            }

            // we follow https://math.stackexchange.com/questions/2713061/jacobian-of-a-quaternion-rotation-wrt-the-quaternion to convert
            // a quaternion to matrix representation for quaternion product
            // i.e., q * p = Q(q) * p
            //       q * p = Qhat(p) * q
            __CPU_AND_CUDA_CODE__ void leftMatrix(double *Q) const
            {
                Q[0] = w;
                Q[1] = -z;
                Q[2] = y;
                Q[3] = x;

                Q[4] = z;
                Q[5] = w;
                Q[6] = -x;
                Q[7] = y;

                Q[8] = -y;
                Q[9] = x;
                Q[10] = w;
                Q[11] = z;

                Q[12] = -x;
                Q[13] = -y;
                Q[14] = -z;
                Q[15] = w;
            }

            __CPU_AND_CUDA_CODE__ void rightMatrix(double *Qhat)
            {
                Qhat[0] = w;
                Qhat[1] = z;
                Qhat[2] = -y;
                Qhat[3] = x;

                Qhat[4] = -z;
                Qhat[5] = w;
                Qhat[6] = x;
                Qhat[7] = y;

                Qhat[8] = y;
                Qhat[9] = -x;
                Qhat[10] = w;
                Qhat[11] = z;

                Qhat[12] = -x;
                Qhat[13] = -y;
                Qhat[14] = -z;
                Qhat[15] = w;
            }
        };
    } // namespace Core
} // namespace SLAM

#endif