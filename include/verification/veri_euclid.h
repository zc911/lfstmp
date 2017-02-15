#ifndef _dgfacesdk_verification_euclid_h_
#define _dgfacesdk_verification_euclid_h_
#include <verification.h>
namespace DGFace {
class EuclidVerification : public Verification {
    public:
        EuclidVerification(void);
        virtual ~EuclidVerification(void);
        float verify(const FeatureType &feature1, const FeatureType &feature2);
    private:
        float score_normalize(float euclid_dist);
};
}
#endif
