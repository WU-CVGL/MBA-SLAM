#ifndef SLAM_CORE_LEVENBERG_MARQUARDT_STRATEGY_H_
#define SLAM_CORE_LEVENBERG_MARQUARDT_STRATEGY_H_

namespace SLAM
{
    namespace VO
    {
        class LevenbergMarquardtStrategy
        {
        public:
            LevenbergMarquardtStrategy();
            ~LevenbergMarquardtStrategy();

        public:
            void reset();
            void step_accepted(double step_quality);
            void step_rejected();

        public:
            double get_radius();

        private:
            double mRadius;
            double mMaxRadius;
            double mMinRadius;
            double mDecreaseFactor;
        };
    } // namespace VO
} // namespace SLAM

#endif