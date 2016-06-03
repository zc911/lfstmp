# Matrix Apps
### Version 0.2.0
2016-05-03

```
GRPC and Restful works at the same time
```

### Modules version
| *Matix Engine* | *Matrix Util* | *Vehicle Model* | *Face Model* |
|:--------------:|:-------------:|:---------------:|:------------:|
| 0.1.5 | 0.1.3 | * | * |

### Features
- Grpc and Restful service are able to works at the same time. The port number of Grpc service
is $PORT(6502) and that of Restful service is $PORT+1(6503)
- Modify Grpc asyn queue from offical to ours (EnginePool)
- Both Grpc and Restful using the same engine pool
- Implements ranker candidates limits
- DEEPVIDEO-305 Matrix - ROI支持(Intreast Areas)
- DEEPVIDEO-306 Matrix - Ranker limitation
- Fix lots of bugs and code improvements

### Bug Fix
- Fix DEEPVIDEO-297 recognize接口中Type字段无效
- Fix DEEPVIDEO-299 ranker接口中type为0和3的时候提示错误
- Fix DEEPVIDEO-300 ranker接口中type为2的时候服务挂掉，无法测试人脸ranker
- Fix DEEPVIDEO-313 uri方式压测gRPC matrix BatchRecognize接口，10-20秒后core dumped
- Fix DEEPVIDEO-315 rpc压测witness接口偶现core
- Fix DEEPVIDEO-316 压测gRPC matrix BatchRecognize接口，cudnn_conv_layer.cu处core dumped

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
//  "ProtocolType": "rpc|restful",
  "ProtocolType": "rpc",
//"ProtocolType": "restful",
  "InstanceType": "witness",
//"InstanceType" : "ranker",
  "System": {
    "Ip": "0.0.0.0",
    "Port": 6502, 
    "GpuNum": 1,
    "ThreadsPerGpu": 7,
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