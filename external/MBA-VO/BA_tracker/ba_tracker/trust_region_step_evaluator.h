#ifndef SLAM_VO_TRUST_REGION_STEP_EVALUATOR_H_
#define SLAM_VO_TRUST_REGION_STEP_EVALUATOR_H_

namespace SLAM
{
    namespace VO
    {
        class TrustRegionStepEvaluator
        {
        public:
            // initial_cost is as the name implies the cost of the starting
            // state of the trust region minimizer.
            //
            // max_consecutive_nonmonotonic_steps controls the window size used
            // by the step selection algorithm to accept non-monotonic
            // steps. Setting this parameter to zero, recovers the classic
            // monotonic descent algorithm.
            TrustRegionStepEvaluator(int max_consecutive_nonmonotonic_steps);

            void reset(double initial_cost);

            // Return the quality of the step given its cost and the decrease in
            // the cost of the model. model_cost_change has to be positive.
            double StepQuality(double cost, double model_cost_change) const;

            // Inform the step evaluator that a step with the given cost and
            // model_cost_change has been accepted by the trust region
            // minimizer.
            void StepAccepted(double cost, double model_cost_change);

        private:
            const int max_consecutive_nonmonotonic_steps_;
            // The minimum cost encountered up till now.
            double minimum_cost_;
            // The current cost of the trust region minimizer as informed by the
            // last call to StepAccepted.
            double current_cost_;
            double reference_cost_;
            double candidate_cost_;
            // Accumulated model cost since the last time the reference model
            // cost was updated, i.e., when a step with cost less than the
            // current known minimum cost is accepted.
            double accumulated_reference_model_cost_change_;
            // Accumulated model cost since the last time the candidate model
            // cost was updated, i.e., a non-monotonic step was taken with a
            // cost that was greater than the current candidate cost.
            double accumulated_candidate_model_cost_change_;
            // Number of steps taken since the last time minimum_cost was updated.
            int num_consecutive_nonmonotonic_steps_;
        };
    } // namespace VO
} // namespace SLAM

#endif // SLAM_VO_TRUST_REGION_STEP_EVALUATOR_H_
