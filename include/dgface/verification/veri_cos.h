#ifndef _dgfacesdk_verification_cos_h_
#define _dgfacesdk_verification_cos_h_
#include <verification.h>
namespace DGFace{

class CosVerification : public Verification {
    public:
        CosVerification(void);
        virtual ~CosVerification(void);
        float verify(const FeatureType &feature1, const FeatureType &feature2);
};
}
#endif
