//
// Created by lzzhao on 12/26/23.
//

#ifndef BA_TRACKER_IMAGE_FACTORY_H
#define BA_TRACKER_IMAGE_FACTORY_H

#include "common.h"

namespace pymbavo {

class Factory {
public:
    Factory() = default;
    ~Factory() = default;
    static ImageU8::Ptr GetImageU8FromTensor(Tensor Image);
    static Eigen::MatrixX2f GetKeypointsfromFrame(Frame::Ptr frame, int camera_id, int level);
};

}  // pymbavo

#endif //BA_TRACKER_IMAGE_FACTORY_H
