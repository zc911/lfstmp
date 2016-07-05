
/*
 * car_feature_extractor.h
 *
 *  Created on: Apr 27, 2016
 *      Author: chenzhen
 */

#ifndef CAR_FEATURE_EXTRACTOR_H_
#define CAR_FEATURE_EXTRACTOR_H_

#include <opencv2/core/core.hpp>

#include "model/rank_feature.h"
#include <opencv2/features2d/features2d.hpp>
namespace dg {

class CarFeatureExtractor {

public:
    CarFeatureExtractor();
    void ExtractDescriptor(const cv::Mat &img, CarRankFeature &des);

private:
    cv::ORB orb_;
    int max_resize_size_;

    void calcNewSize(const ushort &ori_height, const ushort &ori_width,
                     cv::Size &new_size);
};

}
#endif /* CAR_FEATURE_EXTRACTOR_H_ */
