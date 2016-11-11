#ifndef _dgfacesdk_quality_posem_h_
#define _dgfacesdk_quality_posem_h_

#include <quality.h>
#include <headPose.h>
#include <face_inf.h>

namespace DGFace {
class PoseQuality : public Quality {
    public:
	    PoseQuality(void);
		virtual ~PoseQuality(void);
		std::vector<float> quality(const AlignResult &align_result);
};
}
#endif
