#ifndef _dgfacesdk_verification_neuclid_h_
#define _dgfacesdk_verification_neuclid_h_
#include <verification.h>
namespace DGFace {
class NormEuclidVerification : public Verification {
    public:
        NormEuclidVerification(void);
        virtual ~NormEuclidVerification(void);
        float verify(const FeatureType &feature1, const FeatureType &feature2);
};
}
#endif
