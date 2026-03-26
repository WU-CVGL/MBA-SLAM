#ifndef SLAM_CORE_SPLINES_FUNCTOR_H
#define SLAM_CORE_SPLINES_FUNCTOR_H

#include "Matrix.h"
#include "Quaternion.h"
#include "SmallBlas.h"
#include "Vector.h"
#include <cstring>
namespace SLAM
{
    namespace Core
    {
        inline __CPU_AND_CUDA_CODE__ void
        SplineSegmentStartKnotIdxAndNormalizedU(double t, double ctrlKnot_t0, double ctrlKnotSampFreq, int &start_indx, double &u)
        {
            double t_normalized = (t - ctrlKnot_t0) / ctrlKnotSampFreq;
            start_indx = (int)t_normalized;
            u = t_normalized - start_indx;
        }

        inline __CPU_AND_CUDA_CODE__ Vector3d
        C2SplineVec3Functor(const double *data_knots, double u, double *jacobian = nullptr)
        {
            const double one_minus_u = 1 - u;
            Vector3d P;
            P.values[0] = one_minus_u * data_knots[0] + u * data_knots[3];
            P.values[1] = one_minus_u * data_knots[1] + u * data_knots[4];
            P.values[2] = one_minus_u * data_knots[2] + u * data_knots[5];

            if (jacobian != nullptr)
            {
                // jacobian is a 3 x 6 matrix
                std::memset(jacobian, 0, sizeof(double) * 18);
                jacobian[0] = one_minus_u;  //[0][0]
                jacobian[3] = u;            //[0][3]
                jacobian[7] = one_minus_u;  //[1][1]
                jacobian[10] = u;           //[1][4]
                jacobian[14] = one_minus_u; //[2][2]
                jacobian[17] = u;           //[2][5]
            }
            return P;
        }

        inline __CPU_AND_CUDA_CODE__ Vector3d
        C4SplineVec3Functor(const double *data_knots, double u, double *jacobian = nullptr)
        {
            const double uu = u * u;
            const double uuu = uu * u;
            const double one_over_six = 1. / 6.;

            const double coeff0 = one_over_six - 0.5 * u + 0.5 * uu - one_over_six * uuu;
            const double coeff1 = 4 * one_over_six - uu + 0.5 * uuu;
            const double coeff2 = one_over_six + 0.5 * u + 0.5 * uu - 0.5 * uuu;
            const double coeff3 = one_over_six * uuu;

            Vector3d P;
            P.values[0] = coeff0 * data_knots[0] +
                          coeff1 * data_knots[3] +
                          coeff2 * data_knots[6] +
                          coeff3 * data_knots[9];

            P.values[1] = coeff0 * data_knots[1] +
                          coeff1 * data_knots[4] +
                          coeff2 * data_knots[7] +
                          coeff3 * data_knots[10];

            P.values[2] = coeff0 * data_knots[2] +
                          coeff1 * data_knots[5] +
                          coeff2 * data_knots[8] +
                          coeff3 * data_knots[11];

            // compute the jacobian of position with respect to the corresponding control knots
            if (jacobian != nullptr)
            {
                // jacobian is 3 x 12 matrix
                std::memset(jacobian, 0, sizeof(double) * 36);
                jacobian[0] = coeff0;
                jacobian[3] = coeff1;
                jacobian[6] = coeff2;
                jacobian[9] = coeff3;

                jacobian[13] = coeff0;
                jacobian[16] = coeff1;
                jacobian[19] = coeff2;
                jacobian[22] = coeff3;

                jacobian[26] = coeff0;
                jacobian[29] = coeff1;
                jacobian[32] = coeff2;
                jacobian[35] = coeff3;
            }

            return P;
        }

#define MAT44_X_M43(X, Y) \
    Y[0] = X[0] * 0.5;    \
    Y[1] = X[1] * 0.5;    \
    Y[2] = X[2] * 0.5;    \
                          \
    Y[3] = X[4] * 0.5;    \
    Y[4] = X[5] * 0.5;    \
    Y[5] = X[6] * 0.5;    \
                          \
    Y[6] = X[8] * 0.5;    \
    Y[7] = X[9] * 0.5;    \
    Y[8] = X[10] * 0.5;   \
                          \
    Y[9] = X[12] * 0.5;   \
    Y[10] = X[13] * 0.5;  \
    Y[11] = X[14] * 0.5;

#define MAT44_X_K44(X, Y) \
    Y[0] = -X[0];         \
    Y[1] = -X[1];         \
    Y[2] = -X[2];         \
    Y[3] = X[3];          \
                          \
    Y[4] = -X[4];         \
    Y[5] = -X[5];         \
    Y[6] = -X[6];         \
    Y[7] = X[7];          \
                          \
    Y[8] = -X[8];         \
    Y[9] = -X[9];         \
    Y[10] = -X[10];       \
    Y[11] = X[11];        \
                          \
    Y[12] = -X[12];       \
    Y[13] = -X[13];       \
    Y[14] = -X[14];       \
    Y[15] = X[15];

#define MAT44_X_SCALAR(X, lamda) \
    X[0] = X[0] * lamda;         \
    X[1] = X[1] * lamda;         \
    X[2] = X[2] * lamda;         \
    X[3] = X[3] * lamda;         \
                                 \
    X[4] = X[4] * lamda;         \
    X[5] = X[5] * lamda;         \
    X[6] = X[6] * lamda;         \
    X[7] = X[7] * lamda;         \
                                 \
    X[8] = X[8] * lamda;         \
    X[9] = X[9] * lamda;         \
    X[10] = X[10] * lamda;       \
    X[11] = X[11] * lamda;       \
                                 \
    X[12] = X[12] * lamda;       \
    X[13] = X[13] * lamda;       \
    X[14] = X[14] * lamda;       \
    X[15] = X[15] * lamda;

        inline Quaterniond __CPU_AND_CUDA_CODE__
        C2SplineRot3Functor(const double *data_knots,
                            double u,
                            double *jacobian_4x6 = nullptr,
                            double *jacobian_log_exp_2x12 = nullptr,
                            double *X_4x4 = nullptr,
                            double *Y_4x4 = nullptr,
                            double *Z_4x4 = nullptr)
        {
            Quaterniond R0(data_knots[0], data_knots[1], data_knots[2], data_knots[3]);
            Quaterniond R1(data_knots[4], data_knots[5], data_knots[6], data_knots[7]);
            Quaterniond R01 = R0.conjugate() * R1;

            double *domega01_dR01 = nullptr;
            double *dA0_domega01 = nullptr;

            if (jacobian_log_exp_2x12 != nullptr)
            {
                domega01_dR01 = jacobian_log_exp_2x12;
                dA0_domega01 = jacobian_log_exp_2x12 + 12;
            }

            Vector3d omega01 = R01.log(domega01_dR01) * u;
            Quaterniond A0 = Quaterniond::exp(omega01, dA0_domega01);

            if (jacobian_4x6 != nullptr)
            {
                // jacobian is a 4 x 6 matrix, we use local parameterization for quaternion rotation
                std::memset(jacobian_4x6, 0, sizeof(double) * 24);

                // jacobian to R0 local params
                R0.leftMatrix(X_4x4);
                MAT44_X_M43(X_4x4, Y_4x4);
                A0.rightMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 1>(X_4x4, 4, 4, Y_4x4, 4, 3, jacobian_4x6, 0, 0, 0, 6);

                //
                R1.rightMatrix(Z_4x4);
                MAT44_X_K44(Z_4x4, X_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(X_4x4, 4, 4, Y_4x4, 4, 3, Z_4x4, 0, 0, 4, 3);

                MatrixMatrixMultiply<double, double, double, 0>(domega01_dR01, 3, 4, Z_4x4, 4, 3, X_4x4, 0, 0, 3, 3);
                MAT44_X_SCALAR(X_4x4, u);

                MatrixMatrixMultiply<double, double, double, 0>(dA0_domega01, 4, 3, X_4x4, 3, 3, Y_4x4, 0, 0, 4, 3);

                R0.leftMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 1>(X_4x4, 4, 4, Y_4x4, 4, 3, jacobian_4x6, 0, 0, 0, 6);

                // jacobian to R1 local params
                R1.leftMatrix(X_4x4);
                MAT44_X_M43(X_4x4, Y_4x4);
                R0.conjugate().leftMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(X_4x4, 4, 4, Y_4x4, 4, 3, Z_4x4, 0, 0, 4, 3);
                MatrixMatrixMultiply<double, double, double, 0>(domega01_dR01, 3, 4, Z_4x4, 4, 3, X_4x4, 0, 0, 3, 3);
                MAT44_X_SCALAR(X_4x4, u);
                MatrixMatrixMultiply<double, double, double, 0>(dA0_domega01, 4, 3, X_4x4, 3, 3, Y_4x4, 0, 0, 4, 3);
                R0.leftMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 1>(X_4x4, 4, 4, Y_4x4, 4, 3, jacobian_4x6, 0, 3, 0, 6);
            }

            return R0 * A0;
        }

        inline Quaterniond __CPU_AND_CUDA_CODE__
        C4SplineRot3Functor(const double *data_knots,
                            double u,
                            double *jacobian_4x12 = nullptr,
                            double *jacobian_log_exp_6x12 = nullptr,
                            double *X_4x4 = nullptr,
                            double *Y_4x4 = nullptr,
                            double *Z_4x4 = nullptr)
        {
            const double uu = u * u;
            const double uuu = uu * u;
            const double one_over_six = 1. / 6.;

            const double coeff1 = 5 * one_over_six + 0.5 * u - 0.5 * uu + one_over_six * uuu;
            const double coeff2 = one_over_six + 0.5 * u + 0.5 * uu - 2 * one_over_six * uuu;
            const double coeff3 = one_over_six * uuu;

            // qx qy qz qw
            Quaterniond R0(data_knots[0], data_knots[1], data_knots[2], data_knots[3]);
            Quaterniond R1(data_knots[4], data_knots[5], data_knots[6], data_knots[7]);
            Quaterniond R2(data_knots[8], data_knots[9], data_knots[10], data_knots[11]);
            Quaterniond R3(data_knots[12], data_knots[13], data_knots[14], data_knots[15]);

            //
            Quaterniond R01 = R0.conjugate() * R1;
            Quaterniond R12 = R1.conjugate() * R2;
            Quaterniond R23 = R2.conjugate() * R3;

            //
            double *domega01_dR01 = nullptr;
            double *domega12_dR12 = nullptr;
            double *domega23_dR23 = nullptr;
            double *dA0_domega01 = nullptr;
            double *dA1_domega12 = nullptr;
            double *dA2_domega23 = nullptr;

            if (jacobian_log_exp_6x12 != nullptr)
            {
                domega01_dR01 = jacobian_log_exp_6x12;
                domega12_dR12 = jacobian_log_exp_6x12 + 12;
                domega23_dR23 = jacobian_log_exp_6x12 + 24;

                dA0_domega01 = jacobian_log_exp_6x12 + 36;
                dA1_domega12 = jacobian_log_exp_6x12 + 48;
                dA2_domega23 = jacobian_log_exp_6x12 + 60;
            }

            Vector3d omega01 = R01.log(domega01_dR01) * coeff1;
            Vector3d omega12 = R12.log(domega12_dR12) * coeff2;
            Vector3d omega23 = R23.log(domega23_dR23) * coeff3;

            Quaterniond A0 = Quaterniond::exp(omega01, dA0_domega01);
            Quaterniond A1 = Quaterniond::exp(omega12, dA1_domega12);
            Quaterniond A2 = Quaterniond::exp(omega23, dA2_domega23);

            if (jacobian_4x12 != nullptr)
            {
                // jacobian is a 4 x 12 matrix, we use local parameterization for quaternion rotation
                std::memset(jacobian_4x12, 0, sizeof(double) * 48);

                // jacobian to R0 local params
                R0.leftMatrix(X_4x4);
                MAT44_X_M43(X_4x4, Y_4x4);
                (A0 * A1 * A2).rightMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 1>(X_4x4, 4, 4, Y_4x4, 4, 3, jacobian_4x12, 0, 0, 0, 12);

                //
                R1.rightMatrix(Z_4x4);
                MAT44_X_K44(Z_4x4, X_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(X_4x4, 4, 4, Y_4x4, 4, 3, Z_4x4, 0, 0, 4, 3);

                MatrixMatrixMultiply<double, double, double, 0>(domega01_dR01, 3, 4, Z_4x4, 4, 3, X_4x4, 0, 0, 3, 3);
                MAT44_X_SCALAR(X_4x4, coeff1);

                MatrixMatrixMultiply<double, double, double, 0>(dA0_domega01, 4, 3, X_4x4, 3, 3, Y_4x4, 0, 0, 4, 3);

                (A1 * A2).rightMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(X_4x4, 4, 4, Y_4x4, 4, 3, Z_4x4, 0, 0, 4, 3);

                R0.leftMatrix(Y_4x4);
                MatrixMatrixMultiply<double, double, double, 1>(Y_4x4, 4, 4, Z_4x4, 4, 3, jacobian_4x12, 0, 0, 0, 12);

                // jacobian to R1 local params
                R1.leftMatrix(X_4x4);
                MAT44_X_M43(X_4x4, Y_4x4);
                R0.conjugate().leftMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(X_4x4, 4, 4, Y_4x4, 4, 3, Z_4x4, 0, 0, 4, 3);
                MatrixMatrixMultiply<double, double, double, 0>(domega01_dR01, 3, 4, Z_4x4, 4, 3, X_4x4, 0, 0, 3, 3);
                MAT44_X_SCALAR(X_4x4, coeff1);
                MatrixMatrixMultiply<double, double, double, 0>(dA0_domega01, 4, 3, X_4x4, 3, 3, Y_4x4, 0, 0, 4, 3);
                (A1 * A2).rightMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(X_4x4, 4, 4, Y_4x4, 4, 3, Z_4x4, 0, 0, 4, 3);
                R0.leftMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 1>(X_4x4, 4, 4, Z_4x4, 4, 3, jacobian_4x12, 0, 3, 0, 12);

                //
                R1.leftMatrix(X_4x4);
                MAT44_X_M43(X_4x4, Y_4x4);
                R2.rightMatrix(X_4x4);
                MAT44_X_K44(X_4x4, Z_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(Z_4x4, 4, 4, Y_4x4, 4, 3, X_4x4, 0, 0, 4, 3);
                MatrixMatrixMultiply<double, double, double, 0>(domega12_dR12, 3, 4, X_4x4, 4, 3, Y_4x4, 0, 0, 3, 3);
                MAT44_X_SCALAR(Y_4x4, coeff2);
                MatrixMatrixMultiply<double, double, double, 0>(dA1_domega12, 4, 3, Y_4x4, 3, 3, Z_4x4, 0, 0, 4, 3);
                A2.rightMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(X_4x4, 4, 4, Z_4x4, 4, 3, Y_4x4, 0, 0, 4, 3);
                (R0 * A0).leftMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 1>(X_4x4, 4, 4, Y_4x4, 4, 3, jacobian_4x12, 0, 3, 0, 12);

                // jacobian to R2 local params
                R2.leftMatrix(X_4x4);
                MAT44_X_M43(X_4x4, Y_4x4);
                R1.conjugate().leftMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(X_4x4, 4, 4, Y_4x4, 4, 3, Z_4x4, 0, 0, 4, 3);
                MatrixMatrixMultiply<double, double, double, 0>(domega12_dR12, 3, 4, Z_4x4, 4, 3, X_4x4, 0, 0, 3, 3);
                MAT44_X_SCALAR(X_4x4, coeff2);
                MatrixMatrixMultiply<double, double, double, 0>(dA1_domega12, 4, 3, X_4x4, 3, 3, Y_4x4, 0, 0, 4, 3);
                A2.rightMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(X_4x4, 4, 4, Y_4x4, 4, 3, Z_4x4, 0, 0, 4, 3);
                (R0 * A0).leftMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 1>(X_4x4, 4, 4, Z_4x4, 4, 3, jacobian_4x12, 0, 6, 0, 12);

                R2.leftMatrix(X_4x4);
                MAT44_X_M43(X_4x4, Y_4x4);
                R3.rightMatrix(X_4x4);
                MAT44_X_K44(X_4x4, Z_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(Z_4x4, 4, 4, Y_4x4, 4, 3, X_4x4, 0, 0, 4, 3);
                MatrixMatrixMultiply<double, double, double, 0>(domega23_dR23, 3, 4, X_4x4, 4, 3, Y_4x4, 0, 0, 3, 3);
                MatrixMatrixMultiply<double, double, double, 0>(dA2_domega23, 4, 3, Y_4x4, 3, 3, X_4x4, 0, 0, 4, 3);
                MAT44_X_SCALAR(X_4x4, coeff3);
                (R0 * A0 * A1).leftMatrix(Y_4x4);
                MatrixMatrixMultiply<double, double, double, 1>(Y_4x4, 4, 4, X_4x4, 4, 3, jacobian_4x12, 0, 6, 0, 12);

                // jacobian to R3 local params
                R3.leftMatrix(X_4x4);
                MAT44_X_M43(X_4x4, Y_4x4);
                R2.conjugate().leftMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 0>(X_4x4, 4, 4, Y_4x4, 4, 3, Z_4x4, 0, 0, 4, 3);
                MatrixMatrixMultiply<double, double, double, 0>(domega23_dR23, 3, 4, Z_4x4, 4, 3, X_4x4, 0, 0, 3, 3);
                MatrixMatrixMultiply<double, double, double, 0>(dA2_domega23, 4, 3, X_4x4, 3, 3, Z_4x4, 0, 0, 4, 3);
                MAT44_X_SCALAR(Z_4x4, coeff3);
                (R0 * A0 * A1).leftMatrix(X_4x4);
                MatrixMatrixMultiply<double, double, double, 1>(X_4x4, 4, 4, Z_4x4, 4, 3, jacobian_4x12, 0, 9, 0, 12);
            }

            return R0 * A0 * A1 * A2;
        }
#undef MAT44_X_MAT43
#undef MAT44_X_K44
    } // namespace Core
} // namespace SLAM

#endif