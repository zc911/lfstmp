#ifndef _dgfacesdk_recognition_cdnn_h_
#define _dgfacesdk_recognition_cdnn_h_
#include <string>
#include <recognition.h>
#include "Util.h"
#include "ExtractMPMetricFeature.h"

using namespace Util;
using namespace MPMetricFeature;

namespace DGFace{
class CdnnRecog: public Recognition {
	public:
		CdnnRecog(std::string configPath, std::string modelDir);
		virtual ~CdnnRecog();
		void recog_impl(const std::vector<cv::Mat>& faces, 
			const std::vector<AlignResult>& alignment,
			std::vector<RecogResult>& results);
	private:
		// FaceHandler _handler;
		// FacePara _param;
		ExtractMPMetricFeature _extractor;

};
}
#endif
