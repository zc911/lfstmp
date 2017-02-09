#ifndef _DGFACESDK_VERIFICATION_H_
#define _DGFACESDK_VERIFICATION_H_

#include "common.h"
namespace DGFace{

enum class verif_method : unsigned char{
	COS,
	EUCLID,
    NEUCLID
};

class Verification {
    public:
        virtual ~Verification(void) {}
        virtual float verify(const FeatureType &feature1, const FeatureType &feature2) = 0;
    protected:
        Verification(void) {}
};

//Verification *create_verifier(const std::string &prefix = std::string());
Verification *create_verifier(const verif_method& method);
}
#endif

