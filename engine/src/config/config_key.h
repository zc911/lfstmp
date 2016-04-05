/*
 * config_key.h
 *
 *  Created on: 01/04/2016
 *      Author: chenzhen
 */

#ifndef CONFIG_KEY_H_
#define CONFIG_KEY_H_

namespace deepglint {

const string ADDR_VIDEO = "Source";
const string ADDR_VIDEO_ID = "SourceId";
const string ADDR_DEEPV = "DeepV";
const string ADDR_EVENT = "Event";

const string SYS_BUFFER_SIZE = "Sys/BufferSize";
const string SYS_DETECTION_INTERVAL = "Sys/DetectionInterval";
const string SYS_CLASSIFY_INTERVAL = "Sys/ClassifyInterval";
const string SYS_CLASSIFY_BUFFER_SIZE = "Sys/ClassifyBufferSize";
const string VIDEO_INPUT_FPS = "VideoOutput/Fps";
const string VIDEO_INPUT_WIDTH = "VideoOutput/Width";
const string VIDEO_INPUT_HEIGHT = "VideoOutput/Height";
const string VIDEO_PROCESS_FPS = "VideoProcess/Fps";
const string VIDEO_PROCESS_WIDTH = "VideoProcess/Width";
const string VIDEO_PROCESS_HEIGHT = "VideoProcess/Height";
const string VIDEO_OUTPUT_ADDR = "VideoOutput/Addr";
const string VIDEO_OUTPUT_FPS = "VideoOutput/Fps";
const string VIDEO_OUTPUT_WIDTH = "VideoOutput/Width";
const string VIDEO_OUTPUT_HEIGHT = "VideoOutput/Height";
const string VIDEO_OUTPUT_SNAP_WIDTH = "VideoOutput/SnapWidth";
const string VIDEO_OUTPUT_ENABLE = "VideoOutput/Enable";
const string VIDEO_OUTPUT_SHOW_HOTSPOT = "VideoOutput/ShowHotSpot";

const string FEATURE_TYPE = "Feature/Type";
const string FEATURE_HOTSPOTS = "Feature/Hotspots";
const string FEATURE_CAR_TRACKING = "Feature/Car/Tracking";
const string FEATURE_CAR_CLASSIFY = "Feature/Car/Classify";
const string FEATURE_CAR_STYLE = "Feature/Car/Style";
const string FEATURE_CAR_COLOR = "Feature/Car/Color";
const string FEATURE_CAR_PLATE = "Feature/Car/Plate";

const string HOTSPOT_SIZE = "Hotspots/Size";
const string HOTSPOT_HEIGHT = "Hotspots%d/Height";
const string HOTSPOT_WIDTH = "Hotspots%d/Width";
const string HOTSPOT_X = "Hotspots%d/X";
const string HOTSPOT_Y = "Hotspots%d/Y";

const string VIS_ENABLE = "Vis/Enable";
const string VIS_MODE = "Vis/Mode";
const string VIS_MAIN_WIDTH = "Vis/MainWidth";
const string VIS_MAIN_HEIGHT = "Vis/MainHeight";

}

#endif /* CONFIG_KEY_H_ */
