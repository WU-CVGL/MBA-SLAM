#include "SplineTrajectory.h"
#include <iostream>

namespace SLAM
{
    namespace Utils
    {
        SplineBase::SplineBase(double dt_between_control_knots,
                               int n_contrl_knots_per_segment,
                               double gravity,
                               Eigen::Vector3d bacc,
                               Eigen::Vector3d bgyro)
            : mdtBetwContrlKnots(dt_between_control_knots), mnContrlKnotsPerSegment(n_contrl_knots_per_segment), mGravity(gravity), mBiasAcc(bacc), mBiasGyro(bgyro)
        {
        }

        void SplineBase::insert_control_knot(Core::Transformation &T_b2w)
        {
            mControlKnots.push_back(T_b2w);
        }

        SplineCubic::SplineCubic(double dt_between_control_knots,
                                 double gravity,
                                 Eigen::Vector3d bacc,
                                 Eigen::Vector3d bgyro)
            : SplineBase(dt_between_control_knots, 4, gravity, bacc, bgyro)
        {
            mC4 << 6.0, 0.0, 0.0, 0.0,
                5.0, 3.0, -3.0, 1.0,
                1.0, 3.0, 3.0, -2.0,
                0.0, 0.0, 0.0, 1.0;
            mC4 /= 6.0;
        }

        bool SplineCubic::get_interpolation(double t, Core::Transformation &T_b2w)
        {
            double t_normalized = t / mdtBetwContrlKnots;
            int start_idx = t_normalized;
            double u = t_normalized - start_idx;
            if (start_idx + 4 > mControlKnots.size())
            {
                std::cerr << "Do not have enough control knots to do interpolation at timestamp " << t << " second...\n";
                return false;
            }

            // do interpolation
            double uu = u * u;
            Eigen::Vector4d Bu = mC4 * Eigen::Vector4d(1, u, uu, u * uu);

            Eigen::Matrix4d A[3];

            for (int j = 1; j <= 3; j++)
            {
                Core::Transformation T_prev2wld = mControlKnots[start_idx + j - 1];
                Core::Transformation T_curr2wld = mControlKnots[start_idx + j];
                Core::Transformation dT_curr2prev = T_prev2wld.inverse() * T_curr2wld;

                Eigen::Matrix<double, 6, 1> omega = Core::Transformation::log(dT_curr2prev);
                Eigen::Matrix4d omega_hat = Core::Transformation::hat(omega);

                Eigen::Matrix<double, 6, 1> omega_Buj = omega * Bu(j);
                A[j - 1] = Core::Transformation::exp(omega_Buj).matrix();
            }

            // get pose
            Core::Transformation pose0 = mControlKnots[start_idx];
            Eigen::Matrix4d Tm = pose0.matrix() * A[0] * A[1] * A[2];
            T_b2w = Core::Transformation(Tm);
        }

        bool SplineCubic::get_interpolation(double t,
                                            Core::Transformation &T,
                                            Eigen::Vector3d &velocity,
                                            Eigen::Vector3d &gyro,
                                            Eigen::Vector3d &acc)
        {
            double t_normalized = t / mdtBetwContrlKnots;
            int start_idx = t_normalized;
            double u = t_normalized - start_idx;
            if (start_idx + 4 > mControlKnots.size())
            {
                std::cerr << "Do not have enough control knots to do interpolation at timestamp " << t << " second...\n";
                return false;
            }

            // do interpolation
            double uu = u * u;
            double idt = 1.0 / mdtBetwContrlKnots;
            Eigen::Vector4d Bu = mC4 * Eigen::Vector4d(1, u, uu, u * uu);
            Eigen::Vector4d Bud = idt * mC4 * Eigen::Vector4d(0.0, 1.0, 2.0 * u, 3.0 * uu);
            Eigen::Vector4d Budd = (idt * idt) * mC4 * Eigen::Vector4d(0.0, 0.0, 2.0, 6.0 * u);

            Eigen::Matrix4d A[3];
            Eigen::Matrix4d dA[3];
            Eigen::Matrix4d dAA[3];

            for (int j = 1; j <= 3; j++)
            {
                Core::Transformation T_prev2wld = mControlKnots[start_idx + j - 1];
                Core::Transformation T_curr2wld = mControlKnots[start_idx + j];
                Core::Transformation dT_curr2prev = T_prev2wld.inverse() * T_curr2wld;

                Eigen::Matrix<double, 6, 1> omega = Core::Transformation::log(dT_curr2prev);
                Eigen::Matrix4d omega_hat = Core::Transformation::hat(omega);

                Eigen::Matrix<double, 6, 1> omega_Buj = omega * Bu(j);
                A[j - 1] = Core::Transformation::exp(omega_Buj).matrix();
                dA[j - 1] = A[j - 1] * omega_hat * Bud(j);
                dAA[j - 1] = dA[j - 1] * omega_hat * Bud[j] + A[j - 1] * omega_hat * Budd[j];
            }

            // get pose
            Core::Transformation pose0 = mControlKnots[start_idx];
            Eigen::Matrix4d Tm = pose0.matrix() * A[0] * A[1] * A[2];
            T = Core::Transformation(Tm);

            // get acc, gyro
            Eigen::Matrix4d dT = pose0.matrix() * (dA[0] * A[1] * A[2] + A[0] * dA[1] * A[2] + A[0] * A[1] * dA[2]);
            Eigen::Matrix4d dTT = pose0.matrix() * (dAA[0] * A[1] * A[2] + A[0] * dAA[1] * A[2] + A[0] * A[1] * dAA[2] +
                                                    2.0 * (dA[0] * dA[1] * A[2] + dA[0] * A[1] * dA[2] + A[0] * dA[1] * dA[2]));

            Eigen::Matrix3d Rm = Tm.block(0, 0, 3, 3);
            Eigen::Matrix3d Rmt = Rm.transpose();

            Eigen::Matrix3d dR = Rmt * dT.block(0, 0, 3, 3);
            gyro(0) = dR(2, 1);
            gyro(1) = dR(0, 2);
            gyro(2) = dR(1, 0);

            gyro += mBiasGyro;

            velocity = dT.block(0, 3, 3, 1);

            Eigen::Vector3d acc_world = dTT.block(0, 3, 3, 1);
            Eigen::Vector3d acc_worldG = acc_world + Eigen::Vector3d(0.0, 0.0, mGravity);
            acc = Rmt * acc_worldG + mBiasAcc;

            return true;
        }
    } // namespace Utils
} // namespace SLAM