#include "generate_synthetic_data.h"
#include "compute_pixel_intensity.h"
#include "BA_tracker/core/common/Random.h"
#include "BA_tracker/core/common/Vector.h"
#include <Eigen/Dense>

namespace SLAM
{
    namespace VO
    {
        void synthesize_img_with_rand_shapes(const int H, const int W, cv::Mat &im)
        {
            cv::Scalar bkg_color = cv::Scalar(0);
            cv::Scalar fg_color = cv::Scalar(255);

            // generate white background
            if (im.rows != H && im.cols != W)
            {
                im = cv::Mat(H, W, CV_8UC1, bkg_color);
            }

            // draw rectangles
            {
                cv::Point size0(50, 100);
                cv::Point pt00(300, 50);
                cv::Point pt01(pt00.x + size0.x, pt00.y + size0.y);
                cv::Point rect_points[1][4];
                rect_points[0][0] = pt00;
                rect_points[0][1] = cv::Point(pt00.x + size0.x, pt00.y);
                rect_points[0][2] = cv::Point(pt00.x + size0.x, pt00.y + size0.y);
                rect_points[0][3] = cv::Point(pt00.x, pt00.y + size0.y);
                const cv::Point *ppt[1] = {rect_points[0]};
                int npt[] = {4};
                cv::fillPoly(im, ppt, npt, 1, fg_color, cv::LINE_8);
            }

            //
            {
                cv::Point size1(100, 50);
                cv::Point pt10(250, 200);
                cv::Point pt11(pt10.x + size1.x, pt10.y + size1.y);
                cv::Point rect_points[1][4];
                rect_points[0][0] = pt10;
                rect_points[0][1] = cv::Point(pt10.x + size1.x, pt10.y);
                rect_points[0][2] = cv::Point(pt10.x + size1.x, pt10.y + size1.y);
                rect_points[0][3] = cv::Point(pt10.x, pt10.y + size1.y);
                const cv::Point *ppt[1] = {rect_points[0]};
                int npt[] = {4};
                cv::fillPoly(im, ppt, npt, 1, fg_color, cv::LINE_8);
            }

            //
            {
                cv::Point size1(100, 100);
                cv::Point pt10(400, 300);
                cv::Point pt11(pt10.x + size1.x, pt10.y + size1.y);
                cv::Point rect_points[1][4];
                rect_points[0][0] = pt10;
                rect_points[0][1] = cv::Point(pt10.x + size1.x, pt10.y);
                rect_points[0][2] = cv::Point(pt10.x + size1.x, pt10.y + size1.y);
                rect_points[0][3] = cv::Point(pt10.x, pt10.y + size1.y);
                const cv::Point *ppt[1] = {rect_points[0]};
                int npt[] = {4};
                cv::fillPoly(im, ppt, npt, 1, fg_color, cv::LINE_8);
            }

            //
            {
                cv::Point size1(100, 100);
                cv::Point pt10(500, 50);
                cv::Point pt11(pt10.x + size1.x, pt10.y + size1.y);
                cv::Point rect_points[1][4];
                rect_points[0][0] = pt10;
                rect_points[0][1] = cv::Point(pt10.x + size1.x, pt10.y);
                rect_points[0][2] = cv::Point(pt10.x + size1.x, pt10.y + size1.y);
                rect_points[0][3] = cv::Point(pt10.x, pt10.y + size1.y);
                const cv::Point *ppt[1] = {rect_points[0]};
                int npt[] = {4};
                cv::fillPoly(im, ppt, npt, 1, fg_color, cv::LINE_8);
            }

            //
            {
                cv::Point size1(100, 100);
                cv::Point pt10(250, 300);
                cv::Point pt11(pt10.x + size1.x, pt10.y + size1.y);
                cv::Point rect_points[1][4];
                rect_points[0][0] = pt10;
                rect_points[0][1] = cv::Point(pt10.x + size1.x, pt10.y);
                rect_points[0][2] = cv::Point(pt10.x + size1.x, pt10.y + size1.y);
                rect_points[0][3] = cv::Point(pt10.x, pt10.y + size1.y);
                const cv::Point *ppt[1] = {rect_points[0]};
                int npt[] = {4};
                cv::fillPoly(im, ppt, npt, 1, fg_color, cv::LINE_8);
            }

            // draw triangles
            {
                cv::Point tri_pt0(500, 50);
                cv::Point tri_pt1(400, 150);
                cv::Point tri_pt2(550, 250);

                cv::Point rect_points[1][3];
                rect_points[0][0] = tri_pt0;
                rect_points[0][1] = tri_pt1;
                rect_points[0][2] = tri_pt2;
                const cv::Point *ppt[1] = {rect_points[0]};
                int npt[] = {3};
                cv::fillPoly(im, ppt, npt, 1, fg_color, cv::LINE_8);
            }

            {
                cv::Point tri_pt0(150, 300);
                cv::Point tri_pt1(50, 450);
                cv::Point tri_pt2(250, 400);

                cv::Point rect_points[1][3];
                rect_points[0][0] = tri_pt0;
                rect_points[0][1] = tri_pt1;
                rect_points[0][2] = tri_pt2;
                const cv::Point *ppt[1] = {rect_points[0]};
                int npt[] = {3};
                cv::fillPoly(im, ppt, npt, 1, fg_color, cv::LINE_8);
            }
        }

        void warp_image(const cv::Mat &im_ref, const Eigen::Quaterniond &R_c2r, const Eigen::Vector3d &t_c2r, const double &plane_depth, const Core::VectorX<double, 4> &intrinsics, cv::Mat &im_cur)
        {
            const int H = im_ref.rows;
            const int W = im_ref.cols;

            if (im_cur.rows != H && im_cur.cols != W)
            {
                im_cur = cv::Mat(H, W, CV_8UC1);
            }

            unsigned char *im_cur_data_ptr = im_cur.data;
            for (int r = 0; r < H; ++r)
            {
                for (int c = 0; c < W; ++c, ++im_cur_data_ptr)
                {
                    Core::VectorX<double, 2> cur_xy;
                    cur_xy.values[0] = c;
                    cur_xy.values[1] = r;
                    double intensity = 0;
                    compute_pixel_intensity<double>(im_ref.data, nullptr, H, W, R_c2r.coeffs().data(), t_c2r.data(), plane_depth, intrinsics.values[0], intrinsics.values[1], intrinsics.values[2], intrinsics.values[3], cur_xy, &intensity, nullptr);
                    *im_cur_data_ptr = intensity;
                }
            }
        }

        void synthesize_motion_blurred_img(const cv::Mat &I_ref,
                                           const double &plane_depth,
                                           const Core::VectorX<double, 4> &intrinsics,
                                           Core::SplineSE3 *spline,
                                           const double capture_time,
                                           const double exposure_time,
                                           const int num_samples,
                                           cv::Mat &im_blur)
        {
            const int H = I_ref.rows;
            const int W = I_ref.cols;

            cv::Mat I_internal(H, W, CV_32FC1, cv::Scalar(0));
            cv::Mat I_cur;
            cv::Mat temp;

            for (int i = 0; i < num_samples; i++)
            {
                const double t = capture_time - exposure_time * 0.5 + i * exposure_time / (num_samples - 1);
                Eigen::Quaterniond R_c2r;
                Eigen::Vector3d t_c2r;
                spline->getPose(t, R_c2r, t_c2r);
                warp_image(I_ref, R_c2r, t_c2r, plane_depth, intrinsics, I_cur);
                I_cur.convertTo(temp, CV_32FC1);
                I_internal = I_internal + temp;
            }
            I_internal /= num_samples;
            I_internal.convertTo(im_blur, CV_8UC1);
        }
    } // namespace VO
} // namespace SLAM