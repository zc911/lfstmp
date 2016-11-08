#include <verification/veri_cos.h>
#include <verification/veri_euclid.h>
#include <config.h>
#include <stdexcept>
#include <string>

using namespace std;
namespace DGFace{
CosVerification::CosVerification(void) {
}

CosVerification::~CosVerification(void) {
}

float CosVerification::verify(const FeatureType &feature1, const FeatureType &feature2) {
    //cout << feature1.size() << '\t' << feature2.size() << endl;
    assert(feature1.size() && feature1.size() == feature2.size());
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    for(size_t i = 0; i < feature1.size(); ++i) {
        auto val1 = feature1[i];
        auto val2 = feature2[i];
        dot += val1 * val2;
        denom_a += val1 * val1;
        denom_b += val2 * val2;
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}

Verification *create_verifier(const string &prefix) {
    Config *config = Config::instance();
    string type    = config->GetConfig<string>(prefix + "verifier", "cos");
    if (type == "cos")
        return new CosVerification();
    else if(type == "euclid")
        return new EuclidVerification();
    throw new runtime_error("unknown verifier");
}

///////////////////////////////////////////////////////////////
EuclidVerification::EuclidVerification() {
}

EuclidVerification::~EuclidVerification() {
}

float EuclidVerification::verify(const FeatureType &feature1, const FeatureType &feature2) {
    //cout << feature1.size() << '\t' << feature2.size() << endl;
    assert(feature1.size() && feature1.size() == feature2.size());
    float dot = 0.0;
    for(size_t i = 0; i < feature1.size(); ++i) {
        auto val1 = feature1[i];
        auto val2 = feature2[i];
        dot += (val1 - val2) * (val1 - val2);
    }
    return score_normalize(dot);
}

float EuclidVerification::score_normalize(float euclid_dist) {
     const float alpha = -0.04;
     const float beta = 1;
     float norm_score = alpha * euclid_dist + beta;
     if(norm_score > 1) {
         return 1;
     } else if (norm_score < 0) {
         return 0;
     } else {
         return norm_score;
     }
         
}

}
