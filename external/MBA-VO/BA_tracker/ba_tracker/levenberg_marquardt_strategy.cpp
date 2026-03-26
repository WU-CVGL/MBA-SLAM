#include "levenberg_marquardt_strategy.h"
#include <cmath>
#include <iostream>

namespace SLAM
{
    namespace VO
    {
        LevenbergMarquardtStrategy::LevenbergMarquardtStrategy()
            : mRadius(1e4),
              mMinRadius(10),
              mMaxRadius(1e32),
              mDecreaseFactor(2.0)
        {
        }

        LevenbergMarquardtStrategy::~LevenbergMarquardtStrategy()
        {
        }

        void LevenbergMarquardtStrategy::reset()
        {
            mRadius = 1e4;
            mDecreaseFactor = 2.0;
        }

        void LevenbergMarquardtStrategy::step_accepted(double step_quality)
        {
            mRadius = mRadius / std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * step_quality - 1.0, 3));
            mRadius = std::max(std::min(mMaxRadius, mRadius), mMinRadius);
            mDecreaseFactor = 2.0;
        }

        void LevenbergMarquardtStrategy::step_rejected()
        {
            mRadius = mRadius / mDecreaseFactor;
            mRadius = std::max(std::min(mMaxRadius, mRadius), mMinRadius);
            mDecreaseFactor *= 2.0;
        }

        double LevenbergMarquardtStrategy::get_radius()
        {
            return mRadius;
        }
    } // namespace VO
} // namespace SLAM