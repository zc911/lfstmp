#ifndef _DGFACESDK_VERIFICATION_H_
#define _DGFACESDK_VERIFICATION_H_

#include "common.h"
namespace DGFace{
class Verification {
    public:
        virtual ~Verification(void) {}
        virtual float verify(const FeatureType &feature1, const FeatureType &feature2) = 0;
    protected:
        Verification(void) {}
};

Verification *create_verifier(const std::string &prefix = std::string());
}
#endif

