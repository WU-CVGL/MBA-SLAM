//
// Created by peidong on 2/15/20.
//

#ifndef SLAM_ENUMS_H
#define SLAM_ENUMS_H

enum FeatureType
{
    ENUM_SEMIDENSE,
    ENUM_SPARSE_ORB,
    ENUM_SPARSE_SHI_TOMASI
};

enum CameraType
{
    CAMERA_PINHOE,
    CAMERA_UNIFIED
};

enum MotionStatus
{
    STATIC,
    DYNAMIC,
    UNCERTAIN,
    NEWBORN
};

#endif //SLAM_ENUMS_H
