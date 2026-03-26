#ifndef SLAM_CORE_SPLINES_H
#define SLAM_CORE_SPLINES_H

#include <fstream>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <sophus/so3.hpp>

#include "SplineFunctor.h"


namespace SLAM
{
    namespace Core
    {
        class SplineSE3
        {
        public:
            using Ptr = std::shared_ptr<SplineSE3>;
            SplineSE3()
                : mControlKnotSamplingInterval(0),
                  mSplineStartTime(0),
                  mSplineDegK(4)
            {
            }

            SplineSE3(double start_time, double contrlKnotsSamplingInterval)
                : mControlKnotSamplingInterval(contrlKnotsSamplingInterval),
                  mSplineStartTime(start_time),
                  mSplineDegK(4)
            {
            }

            SplineSE3::Ptr clone()
            {
                SplineSE3::Ptr new_spline = std::make_shared<SplineSE3>(mSplineStartTime, mControlKnotSamplingInterval);
                new_spline->setSplineDegK(mSplineDegK);
                const int num_ctrl_knots = mControlKnotsPosition.size();
                for (int i = 0; i < num_ctrl_knots; i++)
                {
                    new_spline->InsertControlKnot(mControlKnotsOrientation.at(i), mControlKnotsPosition.at(i));
                }
                return new_spline;
            }

            void LoadFromFile(std::string &path_to_ctrl_knots)
            {
                std::ifstream fileReader;
                fileReader.open(path_to_ctrl_knots.c_str(), std::ifstream::in);
                if (!fileReader.is_open())
                {
                    printf("%s fails to load control knots from %s, quit...\n", __FUNCTION__, path_to_ctrl_knots.c_str());
                    std::exit(0);
                }

                double start_time = -1;
                double dt = -1;

                while (!fileReader.eof())
                {
                    std::string line;
                    std::getline(fileReader, line);
                    if (line.find('#') != std::string::npos || line.empty())
                        continue;

                    std::stringstream line_stream(line);

                    double t, x, y, z, qx, qy, qz, qw;
                    line_stream >> t >> x >> y >> z >> qx >> qy >> qz >> qw;

                    Eigen::Quaterniond R_b2w(qw, qx, qy, qz);
                    Eigen::Vector3d t_b2w(x, y, z);

                    if (start_time < 0)
                    {
                        start_time = t;
                        this->InsertControlKnot(R_b2w, t_b2w);
                        continue;
                    }

                    if (dt < 0)
                    {
                        dt = t - start_time;
                        start_time = t;
                        this->setSamplingInterval(dt);
                        this->setStartTime(start_time);
                    }
                    this->InsertControlKnot(R_b2w, t_b2w);
                }
            }

        public:
            void setStartTime(double t0)
            {
                mSplineStartTime = t0;
            }

            void setSamplingInterval(double dt)
            {
                mControlKnotSamplingInterval = dt;
            }

            void setSplineDegK(int degK)
            {
                mSplineDegK = degK;
            }

            double getStartTime() const
            {
                return mSplineStartTime;
            }

            double getSamplingInterval() const
            {
                return mControlKnotSamplingInterval;
            }

            int getSplineDegK() const
            {
                return mSplineDegK;
            }

            size_t getNumCtrlKnots() const
            {
                return mControlKnotsPosition.size();
            }

            double *get_knot_data_t()
            {
                return mControlKnotsPosition[0].data();
            }

            double *get_knot_data_R()
            {
                return mControlKnotsOrientation[0].coeffs().data();
            }

            std::vector<Eigen::Vector3d> &getCtrlKnotst()
            {
                return mControlKnotsPosition;
            }

            std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> &getCtrlKnotsR()
            {
                return mControlKnotsOrientation;
            }

        public:
            void InsertControlKnot(const Eigen::Quaterniond& R_b2w, const Eigen::Vector3d& t_b2w)
            {
                mControlKnotsPosition.push_back(t_b2w);
                mControlKnotsOrientation.push_back(R_b2w);
            }

            void InsertControlKnotPy(double w, double x, double y, double z,
                                   const Eigen::Vector3d& t_b2w)
            {
                const Eigen::Quaterniond R_b2w(w, x, y, z);
                InsertControlKnot(R_b2w, t_b2w);
            }

            void PopFrontControlKnot()
            {
                mControlKnotsPosition.erase(mControlKnotsPosition.begin());
                mControlKnotsOrientation.erase(mControlKnotsOrientation.begin());
                mSplineStartTime += mControlKnotSamplingInterval;
            }

            void Clear()
            {
                mControlKnotsPosition.clear();
                mControlKnotsOrientation.clear();
            }

        public:
            void TransformTo(const Eigen::Quaterniond &R_b2w, const Eigen::Vector3d &t_b2w)
            {
                Eigen::Quaterniond R_b2w_orig = mControlKnotsOrientation.at(0);
                Eigen::Vector3d t_b2w_orig = mControlKnotsPosition.at(0);

                Eigen::Quaterniond R_b2w_target = R_b2w;
                Eigen::Vector3d t_b2w_target = t_b2w;

                Eigen::Quaterniond dR = R_b2w_orig.inverse() * R_b2w_target;
                Eigen::Vector3d dt = R_b2w_orig.inverse() * (t_b2w_target - t_b2w_orig);

                for (int i = 0; i < mControlKnotsPosition.size(); i++)
                {
                    mControlKnotsPosition[i] += mControlKnotsOrientation[i] * dt;
                    mControlKnotsOrientation[i] = mControlKnotsOrientation[i] * dR;
                }
            }

            void TransformTo(double t, const Eigen::Quaterniond &R_b2w, const Eigen::Vector3d &t_b2w)
            {
                Eigen::Quaterniond R_b2w_orig;
                Eigen::Vector3d t_b2w_orig;
                this->getPose(t, R_b2w_orig, t_b2w_orig);

                Eigen::Quaterniond R_b2w_target = R_b2w;
                Eigen::Vector3d t_b2w_target = t_b2w;

                Eigen::Quaterniond dR = R_b2w_orig.inverse() * R_b2w_target;
                Eigen::Vector3d dt = R_b2w_orig.inverse() * (t_b2w_target - t_b2w_orig);

                for (int i = 0; i < mControlKnotsPosition.size(); i++)
                {
                    mControlKnotsPosition[i] += mControlKnotsOrientation[i] * dt;
                    mControlKnotsOrientation[i] = mControlKnotsOrientation[i] * dR;
                }
            }

            void TransformBy(const Eigen::Quaterniond &dR, const Eigen::Vector3d &dt)
            {
                for (int i = 0; i < mControlKnotsPosition.size(); i++)
                {
                    mControlKnotsPosition[i] = dR * mControlKnotsPosition[i] + dt;
                    mControlKnotsOrientation[i] = dR * mControlKnotsOrientation[i];
                }
            }

            void TransformByRight(const Eigen::Quaterniond &dR, const Eigen::Vector3d &dt)
            {
                for (int i = 0; i < mControlKnotsPosition.size(); i++)
                {
                    mControlKnotsPosition[i] = mControlKnotsOrientation[i] * dt + mControlKnotsPosition[i];
                    mControlKnotsOrientation[i] = mControlKnotsOrientation[i] * dR;
                }
            }

        public:
            void getPose(double t, Eigen::Quaterniond &R_b2w, Eigen::Vector3d &t_b2w, double *jacobian_R = nullptr, double *jacobian_t = nullptr)
            {
                int start_index;
                double u;
                SplineSegmentStartKnotIdxAndNormalizedU(t,
                                                        mSplineStartTime,
                                                        mControlKnotSamplingInterval,
                                                        start_index,
                                                        u);
//                printf("%s: capture_time %f, start_index: %d, start_time: %f, interval: %f\n",
//                       __FUNCTION__, t, start_index, mSplineStartTime, mControlKnotSamplingInterval);

                assert(start_index >= 0);
                assert(start_index + mSplineDegK <= mControlKnotsPosition.size());
                assert(start_index + mSplineDegK <= mControlKnotsOrientation.size());

                double *t_data_ptr = mControlKnotsPosition.at(start_index).data();
                double *R_data_ptr = mControlKnotsOrientation.at(start_index).coeffs().data();

                double *jacobian_log_exp_6x12 = nullptr;
                double *X_4x4 = nullptr;
                double *Y_4x4 = nullptr;
                double *Z_4x4 = nullptr;
                if (jacobian_R != nullptr)
                {
                    jacobian_log_exp_6x12 = new double[(mSplineDegK - 1) * 24];
                    X_4x4 = new double[16];
                    Y_4x4 = new double[16];
                    Z_4x4 = new double[16];
                }

                Vector3d t_b2w_internal;
                Quaterniond R_b2w_internal;
                switch (mSplineDegK)
                {
                case 2:
                {
                    t_b2w_internal = C2SplineVec3Functor(t_data_ptr, u, jacobian_t);
                    R_b2w_internal = C2SplineRot3Functor(R_data_ptr, u, jacobian_R, jacobian_log_exp_6x12, X_4x4, Y_4x4, Z_4x4);
                    break;
                }
                case 4:
                {
                    t_b2w_internal = C4SplineVec3Functor(t_data_ptr, u, jacobian_t);
                    R_b2w_internal = C4SplineRot3Functor(R_data_ptr, u, jacobian_R, jacobian_log_exp_6x12, X_4x4, Y_4x4, Z_4x4);
                    break;
                }
                default:
                    assert(false);
                }

                memcpy(t_b2w.data(), t_b2w_internal.values, sizeof(double) * 3);
                R_b2w.x() = R_b2w_internal.x;
                R_b2w.y() = R_b2w_internal.y;
                R_b2w.z() = R_b2w_internal.z;
                R_b2w.w() = R_b2w_internal.w;

                delete[] jacobian_log_exp_6x12;
                delete[] X_4x4;
                delete[] Y_4x4;
                delete[] Z_4x4;
            }

            void UpdateCtrlKnot_t(int start_knot_idx, int num_knots, double *dt)
            {
                double *t = mControlKnotsPosition.at(start_knot_idx).data();
                for (int i = 0; i < num_knots; i++)
                {
                    *(t + 3 * i) += *(dt + 3 * i);
                    *(t + 3 * i + 1) += *(dt + 3 * i + 1);
                    *(t + 3 * i + 2) += *(dt + 3 * i + 2);
                }
            }

            void UpdateCtrlKnot_R(int start_knot_idx, int num_knots, double *dR)
            {
                for (int i = 0; i < num_knots; i++)
                {
                    Eigen::Quaterniond &R = mControlKnotsOrientation.at(start_knot_idx + i);
                    const double omega_x = *(dR + 3 * i);
                    const double omega_y = *(dR + 3 * i + 1);
                    const double omega_z = *(dR + 3 * i + 2);
                    R = R * Sophus::SO3d::exp(Eigen::Vector3d(omega_x, omega_y, omega_z)).unit_quaternion();
                    R.normalize();
                }
            }

            void Plus_t(const double *dt, double *candidate_t)
            {
                double *t = mControlKnotsPosition.at(0).data();
                const int len = mControlKnotsPosition.size() * 3;
                for (int i = 0; i < len; ++i, ++t, ++dt, ++candidate_t)
                {
                    *candidate_t = *t + *dt;
                }
            }

            void Plus_R(const double *dR, double *candidate_R)
            {
                const int num_knots = mControlKnotsPosition.size();
                for (int i = 0; i < num_knots; i++)
                {
                    Eigen::Quaterniond &R = mControlKnotsOrientation.at(i);
                    const double omega_x = *(dR + 3 * i);
                    const double omega_y = *(dR + 3 * i + 1);
                    const double omega_z = *(dR + 3 * i + 2);
                    Eigen::Quaterniond dR_ = Sophus::SO3d::exp(Eigen::Vector3d(omega_x, omega_y, omega_z)).unit_quaternion();
                    Eigen::Quaterniond R_ = R * dR_;
                    memcpy(candidate_R + i * 4, R_.coeffs().data(), sizeof(double) * 4);
                }
            }

            void InvalidParameter(double *data_t, double *data_R)
            {
                memcpy(mControlKnotsPosition[0].data(),
                       data_t,
                       sizeof(double) * 3 * mControlKnotsPosition.size());

                memcpy(mControlKnotsOrientation[0].coeffs().data(),
                       data_R,
                       sizeof(double) * 4 * mControlKnotsOrientation.size());
            }

            void ResetIdentity()
            {
                memset(mControlKnotsPosition[0].data(),
                       0,
                       sizeof(double) * 3 * mControlKnotsPosition.size());

                const int num_knots = mControlKnotsPosition.size();
                for (int i = 0; i < num_knots; i++)
                {
                    Eigen::Quaterniond &R = mControlKnotsOrientation.at(i);
                    R.setIdentity();
                }
            }

        private:
            // The position and orientation is defined from body frame to world frame
            // T = [R, t]
            std::vector<Eigen::Vector3d> mControlKnotsPosition;
            std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> mControlKnotsOrientation;

            // Control knots sampling interval.
            double mControlKnotSamplingInterval;
            double mSplineStartTime;

            //
            int mSplineDegK;
        };
    } // namespace Core
} // namespace SLAM

#endif