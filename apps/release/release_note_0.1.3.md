# Matrix Apps
### Version 0.1.2
2016-05-28

```
Multiple thread(GRPC only)
```

### Modules version
| *Matix Engine* | *Matrix Util* | *Vehicle Model* | *Face Model* |
|:--------------:|:-------------:|:---------------:|:------------:|
| 0.1.2 | 0.1.3 | * | * |

### Features
- Multiple thread processing, one matrix apps processing with more than one matrix engines
- New config items: "GpuNum" and "ThreadsPerGpu"
- A simple thread pool
- Parse images in concurrent with simple thread pool
- Requests check, return error if find invalid inputs
- WenTong Plate SDK works with multiple threads
- Performance improvement
- Some code improement

### Bug Fix
DEEPVIDEO-310 witness.proto:start the field numbering at zero
DEEPVIDEO-302 输入无法访问到的url, Matrix将长时间阻塞, 无法处理后续请求


### How to Install/Update
```
$ wget -O install.sh http://192.168.2.119/matrix/install.sh
$ chmod +x install.sh
$ ./install.sh 
```

### How to Run
```
$ sudo ./matrix_app
```

### Config File
```json
{
  "Version": {
    "Code": "1.0.0",
    "Model": "1.9"
  },
  "ProtocolType": "rpc",
  //"ProtocolType": "restful",
  "InstanceType": "witness",
  //"InstanceType" : "ranker",
  "System": {
    "Ip": "0.0.0.0",
    "Port": 6500,
    "GpuNum": 1, 
    "ThreadsPerGpu": 4,
    "EnableAsyn": true
  },
  "Feature": {
    "Vehicle": {
      "Enable": true,
      "EnableDetection": true,
      "EnableType": true,
      "EnableColor": true,
      "EnablePlate": true,
      "EnablePlateEnhanced": false,
      "EnableMarker": true,
      "EnableFeatureVector": false
    },
    "Face": {
      "Enable": false,
      "EnableDetection": true,
      "EnableFeatureVector": true
    }
  },
  "Advanced": {
    "Detection": {
      "Rescale": 400,
      "BatchSize": 8
    },
    "Color": {
      "BatchSize": 8
    },
    "Style": {
      "BatchSize": 8
    },
    "Plate": {
      "LocalProvince": "",
      "OCR": 1,
      "Locate": 5,
      "EnableSharp": true
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
    }
  },
  "Log": {
    "Dir": "",
    "Level": ""
  },
  "Render": {
    "Vehicle": {
      "Model": "models/mapping/front_day_index_1_8.txt",
      "Color": "models/mapping/vehicle_color.txt",
      "Symbol": "models/mapping/vehicle_symbol.txt",
      "Plate": {
        "Color": "models/mapping/plate_color.txt",
        "Type": "models/mapping/plate_type.txt"
      }
    }
  },
  "Debug": {
    "Enable": true,
    "Encrypt": false
  },
  "DataPath": "data_config.json"
}

```