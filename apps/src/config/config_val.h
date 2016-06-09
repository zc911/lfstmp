/*
 * config_val.h
 *
 *  Created on: Apr 25, 2016
 *      Author: chenzhen
 */

#ifndef CONFIG_VAL_H_
#define CONFIG_VAL_H_

#include <string>

using namespace std;

namespace dg {

static const string VEHICLE_MODEL_MAPPING_FILE = "Render/Vehicle/Model";
static const string VEHICLE_COLOR_MAPPING_FILE = "Render/Vehicle/Color";
static const string VEHICLE_SYMBOL_MAPPING_FILE = "Render/Vehicle/Symbol";
static const string VEHICLE_PLATE_COLOR_MAPPING_FILE =
        "Render/Vehicle/Plate/Color";
static const string VEHICLE_PLATE_TYPE_MAPPING_FILE =
        "Render/Vehicle/Plate/Type";
static const string VEHICLE_TYPE_MAPPING_FILE = "Render/Vehicle/Type";

static const string SYSTEM_GPU_NUM = "System/GpuNum";
static const string SYSTEM_THREADS_PER_GPU = "System/ThreadsPerGpu";
static const string RANKER_DEFAULT_TYPE = "Ranker/DefaultType";
static const string VERSION_MODEL = "Version/Model";
static const string SERVICE_MODEL = "Version/Code";

static const string RENDER_VEHICLE_TYPE = "Render/Vehicle/Type";
static const string RENDER_VEHICLE_MODEL = "Render/Vehicle/Model";
static const string RENDER_VEHICLE_COLOR = "Render/Vehicle/Color";
static const string RENDER_VEHICLE_SYMBOL = "Render/Vehicle/Symbol";
static const string RENDER_VEHICLE_PLATE_COLOR = "Render/Vehicle/Plate/Color";
static const string RENDER_VEHICLE_PLATE_TYPE = "Render/Vehicle/Plate/Type";
static const string STORAGE_ADDRESS = "Storage/Address";
static const string STORAGE_ENABLED = "Storage/Enabled";


}

#endif /* CONFIG_VAL_H_ */
