# Matrix Apps
### Version 0.2.1
2016-06-07

```
VLOG policy and GFlag
```

### Modules version
| *Matix Engine* | *Matrix Util* | *Vehicle Model* | *Face Model* |
|:--------------:|:-------------:|:---------------:|:------------:|
| 0.1.6 | 0.1.3 | * | * |

### Features
- Update grpc to 0.14.1
- Release grpc and protobuf libraries together with the binary
- Move all dependency libraries to libs folder
- Define VLOG policy:
```
# define VLOG_SERVICE  0
# define VLOG_PROCESS_COST  2 
# define VLOG_RUNTIME_DEBUG  8 
```
- Use GFlags as the command line parser, define two flags:
```
 -port=$PORT 
 -config=$CONFIG_FILE_PATH 
```


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
    "Model": "1.9"
  },
  "ProtocolType": "rpc|restful",
//  "ProtocolType": "rpc",
//"ProtocolType": "restful",
  "InstanceType": "witness",
//"InstanceType" : "ranker",
  "System": {
    "Ip": "0.0.0.0",
    "Port": 6502,
    "GpuNum": 1,
    "ThreadsPerGpu": 1,
    "EnableAsyn": true
  },
  "Feature": {
    "Vehicle": {
      "Enable": true,
      "EnableDetection": true,
      "EnableType": true,
      "EnableColor": true,
      "EnablePlate": true, 
      "EnableGpuPlate": false,
      "EnablePlateEnhanced": true,
      "EnableMarker": true, 
      "EnableFeatureVector": true 
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
      "BatchSize": 8,
      "ModelNum": 2
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
      "BatchSize": 30
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
      "Maximum": 1000
    }
  },
  "Log": {
    "Dir": "",
    "Level": ""
  },
  "Render": {
    "Vehicle": {
      "Type": "models/mapping/vehicle_type.txt",
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