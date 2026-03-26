//
// Created by peidong on 2/14/20.
//

#include <fstream>
#include <iostream>
#include "InputOutput.h"
#include "IsClose.h"

namespace SLAM {
    namespace Utils {
        void load_depthMap(std::string path_to_file, SLAM::Core::Image<float>::Ptr depthMap) {
            std::ifstream fileReader;
            fileReader.open(path_to_file);

            if (!fileReader.is_open()) {
                std::cerr << "failed to open " << path_to_file << "\n";
                exit(0);
            }

            // read

            size_t H = depthMap->nHeight();
            size_t W = depthMap->nWidth();
            float *tempDepthMap = new float[H * W];
            float *copyTempDepthMapPtr = tempDepthMap;

            for (int r = 0; r < H; ++r) {
                for (int c = 0; c < W; ++c, ++copyTempDepthMapPtr) {
                    float depth;
                    fileReader >> depth;
                    if (depth > 100) {
                        depth = 0.;
                    }
                    copyTempDepthMapPtr[0] = depth;
                }
            }

            // copy
            depthMap->copyFrom(tempDepthMap, H, W, 1);
            delete[] tempDepthMap;
        }

        void save_pcl_to_plyFile(Core::Image<float> *pcl, std::string save_file_path) {
            std::ofstream fileWriter;
            fileWriter.open(save_file_path.c_str(), std::ofstream::out);
            if (!fileWriter.is_open()) {
                std::cout << "Failed to open " << save_file_path.c_str() << "\n";
            }
            std::cout << "Point cloud is saved to " << save_file_path.c_str() << "\n";

            // write header
            fileWriter << "ply\n";
            fileWriter << "format ascii 1.0\n";
            fileWriter << "element vertex " << pcl->nWidth() * pcl->nHeight() << "\n";
            fileWriter << "property float32 x\n";
            fileWriter << "property float32 y\n";
            fileWriter << "property float32 z\n";
            fileWriter << "end_header\n";

            float *pcl_ptr = pcl->getData();
            for (int r = 0; r < pcl->nHeight(); ++r) {
                for (int c = 0; c < pcl->nWidth(); ++c, pcl_ptr += 3) {
                    fileWriter << pcl_ptr[0] << " " << pcl_ptr[1] << " " << pcl_ptr[2] << "\n";
                }
            }
            fileWriter.close();
        }

        bool unreal_get_ground_truth_nav_state(std::string    path_to_pose,
                                              double          _t,
                                              Core::NavState *navState) {
          std::ifstream fileReader;
          fileReader.open(path_to_pose.c_str(), std::ifstream::in);

          bool found = false;
          while (!found && !fileReader.eof()) {
            std::string line;
            std::getline(fileReader, line);
            if (line.find('#') != std::string::npos)
              continue;

            std::stringstream line_stream(line);

            double t, qx, qy, qz, qw, x, y, z, vx, vy, vz, rx, ry, rz, ax, ay,
                az;
            line_stream >> t >> qx >> qy >> qz >> qw >> x >> y >> z >> vx >>
                vy >> vz >> rx >> ry >> rz >> ax >> ay >> az;

            if (!Utils::is_close<double>(_t, t))
              continue;

            navState->setPose(Core::Transformation(
                Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(x, y, z)));
            navState->setVelocity(Eigen::Vector3d(vx, vy, vz));
            found = true;
          }
          return found;
        }

        bool unreal_get_imu_measurements(std::string path_to_pose, 
                                        double t_start, 
                                        double t_end, 
                                        std::vector<Core::ImuMeasurement> &imuMeasurements) 
        {
          std::ifstream fileReader;
          fileReader.open(path_to_pose.c_str(), std::ifstream::in);

          bool          done = false;
          double        t_prev = 0;
          Core::ImuMeasurement data_prev;

          while (!done && !fileReader.eof()) {
            std::string line;
            std::getline(fileReader, line);
            if (line.find('#') != std::string::npos)
              continue;

            std::stringstream line_stream(line);

            double t, qx, qy, qz, qw, x, y, z, vx, vy, vz, rx, ry, rz, ax, ay,
                az;
            line_stream >> t >> qx >> qy >> qz >> qw >> x >> y >> z >> vx >>
                vy >> vz >> rx >> ry >> rz >> ax >> ay >> az;

            if (t < t_start)
              continue;
            if (Utils::is_close<double>(t, t_end))
              done = true;

            Core::ImuMeasurement data;
            data.acc = Eigen::Vector3d(ax, ay, az);
            data.gyro = Eigen::Vector3d(rx, ry, rz);
            data.timestamp = t;

            if (Utils::is_close<double>(t, t_start)) {
              data_prev = data;
              t_prev = t;
              continue;
            }

            data_prev.dt = (t - t_prev);
            imuMeasurements.push_back(data_prev);

            // std::cout << data_prev.getTimestamp() << " " <<
            // data_prev.getDeltat() << "\n";

            data_prev = data;
            t_prev = t;
          }
          return done;
        }
    }
}