# Matrix Apps
### Version 0.2.6
2016-06-30

### Modules version
| *Matix Engine* | *Matrix Util* | *Vehicle Model* | *Face Model* |
|:--------------:|:-------------:|:---------------:|:------------:|
| 0.1.9 | 0.1.3 | 1.10 | * |

### Features
- Improve performance without ssd detection
- New color models

### Bugs fixed
- DEEPVIDEO-357
- DEEPVIDEO-356
- DEEPVIDEO-355
- DEEPVIDEO-341
- DEEPVIDEO-329
- Cutboard image data is empty

### How to Install/Update
```
$ wget -O install.sh http://192.168.2.119/matrix/install.sh
$ chmod +x install.sh
$ ./install.sh 
```

### How to Run
```
$ sudo ./matrix_app [-port=$PORT] [-config=$CONFIG_FILE_PATH]
or
$ sudo ./matrix_app -help
for more help details 
```

### Config File
```json
{
  "Version": {
    "Code": "1.0.0",
    "Model": "1.10"
  },
//  "ProtocolType": "rpc",
    "ProtocolType": "restful",
 "InstanceType": "witness",
//  "InstanceType" : "ranker",
  "System": {
    "Ip": "0.0.0.0",
    "Port": 6602,
    "GpuNum": 1,
    "ThreadsPerGpu": 1,
    "EnableAsyn": true
  },
  "Ranker":{
	"DefaultType":2
  },
  "Feature": {
    "Vehicle": {
      "Enable": true,
      "EnableDetection":true,
      "EnableType": true,
      "EnableColor":true,
      "EnablePlate": false,
      "EnableGpuPlate": true,
      "EnablePlateEnhanced": false,
      "EnableMarker": true,
      "EnableFeatureVector":true
    },
    "Face": {
      "Enable": false,
      "EnableDetection": false,
      "EnableFeatureVector": false
    }
  },
  "Advanced": {
    "Detection": {
      "Rescale": 400,
      "BatchSize": 1
    },
    "Color": {
      "BatchSize": 1,
      "ModelNum": 1
    },
    "Style": {
      "BatchSize": 8,
      "ModelNum": 2
    },
    "Plate": {
      "LocalProvince": "",
      "OCR": 1,
      "Locate": 5,
      "EnableSharp": false
    },
    "PlateMxnet": {
      "BatchSize": 10
    },
    "Marker": {
      "MOTConfidence": 0.6,
      "BeltConfidence": 0.8,
      "GlobalConfidence": 0.8,
      "AccessoriesConfidence": 0.8,
      "OthersConfidence": 0.8,
      "TissueBoxConfidence": 0.8,
      "SunVisorConfidence": 0.8,
      "BatchSize": 1
    },
    "Window": {
      "BatchSize": 8
    },
    "FaceDetect": {
      "BatchSize": 1,
      "Confidence": 0.7,
      "Scale": 640
    },
    "FaceExtract": {
      "BatchSize": 1
    },
    "Ranker":{
	"Maximum":100000
    }
  },
  "Log": {
    "Dir": "",
    "Level": ""
  },
  "Render": {
    "Vehicle": {
      "Type": "unencrypted_models/mapping/vehicle_type.txt",
      "Model": "unencrypted_models/mapping/front_day_index_1_8.txt",
      "Color": "unencrypted_models/mapping/vehicle_colorhaoquan.txt",
      "Symbol": "unencrypted_models/mapping/vehicle_symbol.txt",
      "Plate": {
        "Color": "unencrypted_models/mapping/plate_color.txt",
	"ColorGpu":"unencrypted_models/mapping/plate_gpu_color.txt",
        "Type": "unencrypted_models/mapping/plate_type.txt"
      }
    }
  },
  "Debug": {
    "Enable": true,
	"EnableCutboard":true,
    "Encrypt": false
  },
  "Storage":{
    "Enabled":true,
    "Address":"192.168.2.119:9004",
    "DBType":2
  },
  "DataPath": "data_config_unencrypted.json"
}

```