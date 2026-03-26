#include "FullSystem/FullSystem.h"

#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "IOWrapper/ImageDisplay.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/CoarseTracker.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

#include "core/common/Time.h"

namespace dso
{
    void FullSystem::addActiveFrame(ImageAndExposure *image, ImageAndExposure *im_sharp, SLAM::Core::Image<float> *depth_map, int id)
    {
        if (isLost)
            return;
        boost::unique_lock<boost::mutex> lock(trackMutex);

        // =========================== add into allFrameHistory =========================
        FrameHessian *fh = new FrameHessian();
        FrameShell *shell = new FrameShell();
        shell->camToWorld = SE3(); // no lock required, as fh is not used anywhere yet.
        shell->aff_g2l = AffLight(0, 0);
        shell->marginalizedAt = shell->id = allFrameHistory.size();
        shell->timestamp = image->timestamp;
        shell->incoming_id = id;
        fh->shell = shell;
        allFrameHistory.push_back(shell);

        // =========================== make Images / derivatives etc. =========================
        fh->ab_exposure = image->exposure_time;
        fh->makeImages(image->image, &Hcalib);

        if (!initialized)
        {
            fh->makeImages(im_sharp->image, &Hcalib);
            // use initializer!
            if (coarseInitializer->frameID < 0) // first frame set. fh is kept by coarseInitializer.
            {

                coarseInitializer->setFirst(&Hcalib, fh, depth_map);
            }
            else if (coarseInitializer->trackFrame(fh, outputWrapper)) // if SNAPPED
            {
                initializeFromInitializer(fh);
                lock.unlock();
                deliverTrackedFrame(fh, true);
            }
            else
            {
                // if still initializing
                fh->shell->poseValid = false;
                delete fh;
            }
            return;
        }
        else // do front-end operation.
        {
            // =========================== SWAP tracking reference?. =========================
            if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
            {
                boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
                CoarseTracker *tmp = coarseTracker;
                coarseTracker = coarseTracker_forNewKF;
                coarseTracker_forNewKF = tmp;
            }

            if (mIsBlurAware)
            {
                bool isKeyframe = trackNewCoarseBlurAware(fh);
                for (IOWrap::Output3DWrapper *ow : outputWrapper)
                    ow->publishCamPose(fh->shell, &Hcalib);

                lock.unlock();

                if (isKeyframe)
                {
                    fh->makeImages(im_sharp->image, &Hcalib);
                }
                deliverTrackedFrame(fh, isKeyframe);
                return;
            }
            else
            {
                Vec4 tres = trackNewCoarse(fh);
                if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
                {
                    printf("Initial Tracking failed: LOST!\n");
                    isLost = true;
                    return;
                }

                bool needToMakeKF = false;
                if (setting_keyframesPerSecond > 0)
                {
                    needToMakeKF = allFrameHistory.size() == 1 ||
                                   (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f / setting_keyframesPerSecond;
                }
                else
                {
                    Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                                               coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

                    // BRIGHTNESS CHECK
                    needToMakeKF = allFrameHistory.size() == 1 ||
                                   setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
                                           setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double)tres[2]) / (wG[0] + hG[0]) +
                                           setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0]) +
                                           setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0])) >
                                       1 ||
                                   2 * coarseTracker->firstCoarseRMSE < tres[0];
                }

                for (IOWrap::Output3DWrapper *ow : outputWrapper)
                    ow->publishCamPose(fh->shell, &Hcalib);

                lock.unlock();

                if (needToMakeKF)
                {
                    fh->makeImages(im_sharp->image, &Hcalib);
                }
                deliverTrackedFrame(fh, needToMakeKF);
                return;
            }
        }
    }
    void FullSystem::deliverTrackedFrame(FrameHessian *fh, bool needKF)
    {

        if (linearizeOperation)
        {
            if (goStepByStep && lastRefStopID != coarseTracker->refFrameID)
            {
                MinimalImageF3 img(wG[0], hG[0], fh->dI);
                IOWrap::displayImage("frameToTrack", &img);
                while (true)
                {
                    char k = IOWrap::waitKey(0);
                    if (k == ' ')
                        break;
                    handleKey(k);
                }
                lastRefStopID = coarseTracker->refFrameID;
            }
            else
                handleKey(IOWrap::waitKey(1));

            if (needKF)
                makeKeyFrame(fh);
            else
                makeNonKeyFrame(fh);
        }
        else
        {
            boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
            unmappedTrackedFrames.push_back(fh);
            if (needKF)
                needNewKFAfter = fh->shell->trackingRef->id;
            trackedFrameSignal.notify_all();

            while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1)
            {
                mappedFrameSignal.wait(lock);
            }

            lock.unlock();
        }
    }

    void FullSystem::initializeTracker(std::string &config_path)
    {
        coarseTracker->initializeBlurAwareTracker(config_path);
        coarseTracker_forNewKF->initializeBlurAwareTracker(config_path);
    }

    Vec4 FullSystem::trackNewCoarse(FrameHessian *fh)
    {

        assert(allFrameHistory.size() > 0);
        // set pose initialization.

        for (IOWrap::Output3DWrapper *ow : outputWrapper)
            ow->pushLiveFrame(fh);

        FrameHessian *lastF = coarseTracker->lastRef;

        AffLight aff_last_2_l = AffLight(0, 0);

        std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
        if (allFrameHistory.size() == 2)
            for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
                lastF_2_fh_tries.push_back(SE3());
        else
        {
            FrameShell *slast = allFrameHistory[allFrameHistory.size() - 2];
            FrameShell *sprelast = allFrameHistory[allFrameHistory.size() - 3];
            SE3 slast_2_sprelast;
            SE3 lastF_2_slast;
            { // lock on global pose consistency!
                boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
                slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
                lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
                aff_last_2_l = slast->aff_g2l;
            }
            SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast.

            // get last delta-movement.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);                        // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast); // assume double motion (frame skipped)
            lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast);  // assume half motion.
            lastF_2_fh_tries.push_back(lastF_2_slast);                                               // assume zero motion.
            lastF_2_fh_tries.push_back(SE3());                                                       // assume zero motion FROM KF.

            // just try a TON of different initializations (all rotations). In the end,
            // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
            // also, if tracking rails here we loose, so we really, really want to avoid that.
            for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++)
            {
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));                  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));                  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));                  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));                 // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));                 // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));                 // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));           // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));           // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));           // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));         // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));         // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));         // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));    // assume constant motion.
            }

            if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
            {
                lastF_2_fh_tries.clear();
                lastF_2_fh_tries.push_back(SE3());
            }
        }

        Vec3 flowVecs = Vec3(100, 100, 100);
        SE3 lastF_2_fh = SE3();
        AffLight aff_g2l = AffLight(0, 0);

        // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
        // I'll keep track of the so-far best achieved residual for each level in achievedRes.
        // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

        Vec5 achievedRes = Vec5::Constant(NAN);
        bool haveOneGood = false;
        int tryIterations = 0;
        for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
        {
            AffLight aff_g2l_this = aff_last_2_l;
            SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

            bool trackingIsGood = coarseTracker->trackNewestCoarse(
                fh, lastF_2_fh_this, aff_g2l_this,
                pyrLevelsUsed - 1,
                achievedRes); // in each level has to be at least as good as the last try.
            tryIterations++;

            if (i != 0)
            {
                printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
                       i,
                       i, pyrLevelsUsed - 1,
                       aff_g2l_this.a, aff_g2l_this.b,
                       achievedRes[0],
                       achievedRes[1],
                       achievedRes[2],
                       achievedRes[3],
                       achievedRes[4],
                       coarseTracker->lastResiduals[0],
                       coarseTracker->lastResiduals[1],
                       coarseTracker->lastResiduals[2],
                       coarseTracker->lastResiduals[3],
                       coarseTracker->lastResiduals[4]);
            }

            // do we have a new winner?
            if (trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >= achievedRes[0]))
            {
                //printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
                flowVecs = coarseTracker->lastFlowIndicators;
                aff_g2l = aff_g2l_this;
                lastF_2_fh = lastF_2_fh_this;
                haveOneGood = true;
            }

            // take over achieved res (always).
            if (haveOneGood)
            {
                for (int i = 0; i < 5; i++)
                {
                    if (!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i]) // take over if achievedRes is either bigger or NAN.
                        achievedRes[i] = coarseTracker->lastResiduals[i];
                }
            }

            if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
                break;
        }

        if (!haveOneGood)
        {
            printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
            flowVecs = Vec3(0, 0, 0);
            aff_g2l = aff_last_2_l;
            lastF_2_fh = lastF_2_fh_tries[0];
        }

        lastCoarseRMSE = achievedRes;

        // no lock required, as fh is not used anywhere yet.
        fh->shell->camToTrackingRef = lastF_2_fh.inverse();
        fh->shell->trackingRef = lastF->shell;
        fh->shell->aff_g2l = aff_g2l;
        fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

        if (coarseTracker->firstCoarseRMSE < 0)
            coarseTracker->firstCoarseRMSE = achievedRes[0];

        if (!setting_debugout_runquiet)
            printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);

        if (setting_logStuff)
        {
            (*coarseTrackingLog) << std::setprecision(16)
                                 << fh->shell->id << " "
                                 << fh->shell->timestamp << " "
                                 << fh->ab_exposure << " "
                                 << fh->shell->camToWorld.log().transpose() << " "
                                 << aff_g2l.a << " "
                                 << aff_g2l.b << " "
                                 << achievedRes[0] << " "
                                 << tryIterations << "\n";
        }

        return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
    }

    bool FullSystem::trackNewCoarseBlurAware(FrameHessian *fh)
    {
        assert(allFrameHistory.size() > 0);
        // set pose initialization.

        for (IOWrap::Output3DWrapper *ow : outputWrapper)
            ow->pushLiveFrame(fh);

        FrameHessian *latest_keyframe = coarseTracker->lastRef;

        // assume constant motion velocity prediction model
        std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
        SE3 lastF_2_fh;
        if (allFrameHistory.size() > 2)
        {
            FrameShell *slast = allFrameHistory[allFrameHistory.size() - 2];
            FrameShell *sprelast = allFrameHistory[allFrameHistory.size() - 3];
            SE3 slast_2_sprelast;
            SE3 lastF_2_slast;
            { // lock on global pose consistency!
                boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
                slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
                lastF_2_slast = slast->camToWorld.inverse() * latest_keyframe->shell->camToWorld;
            }
            SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast.
            lastF_2_fh = (fh_2_slast.inverse() * lastF_2_slast);
        }

        for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta += 0.01)
        {
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));                  // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));                  // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));                  // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));                 // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));                 // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));                 // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));           // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));           // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));           // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));         // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));         // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));         // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
            lastF_2_fh_tries.push_back(lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));    // assume constant motion.
        }

        //
        AffLight aff_g2l = AffLight(0, 0);
        bool is_keyframe = false;
        Vec5 failure_indicators = Vec5::Constant(0);

        SLAM::Core::WallTime start_time = SLAM::Core::currentTime();
        is_keyframe = coarseTracker->trackNewestCoarseBlurAware(
            fh,
            lastF_2_fh,
            aff_g2l,
            pyrLevelsUsed - 1,
            failure_indicators); // in each level has to be at least as good as the last try.
        fh->shell->trackerTimeConsumption_ms = SLAM::Core::elapsedTimeInMilliSeconds(start_time);

        const int max_blur_kernel_len = 96;
        if (failure_indicators[1] > max_blur_kernel_len)
        {
            std::cout << "tracking failed...\n";
            // relocalize
            for (int i = 0; i < lastF_2_fh_tries.size(); i++)
            {
                std::cout << "relocalizing...\n";
                lastF_2_fh = lastF_2_fh_tries[i];
                is_keyframe = coarseTracker->relocalizeBlurAware(fh, lastF_2_fh, failure_indicators);

                if (failure_indicators[1] < max_blur_kernel_len)
                {
                    break;
                }
            }
        }

        // no lock required, as fh is not used anywhere yet.
        fh->shell->camToTrackingRef = lastF_2_fh.inverse();
        fh->shell->trackingRef = latest_keyframe->shell;
        fh->shell->aff_g2l = aff_g2l;
        fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

        return is_keyframe;
    } // namespace dso

} // namespace dso