# Matrix Apps
### Version 0.1.2
2016-05-23

```
gRpc interface works
```

### Modules version
| *Matix Engine* | *Matrix Util* | *Vehicle Model* | *Face Model* |
|:--------------:|:-------------:|:---------------:|:------------:|
| 0.1.1 | 0.1.1 | * | * |

### Features
- gRpc interface works


### How to Install/Update
```
$ wget http://192.168.2.119/matrix/install.sh
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
  // Instance type could be "witness" or "ranker"
  "InstanceType": "witness",
  "System": {
    "Ip": "0.0.0.0",
    "Port": 6500,
    "GpuId": 0,
    "EnableAsyn": true
  },
  "Advanced": {
    "Detection": {
      "Rescale": 400,
      "BatchSize": 1
    },
    "Color": {
      "BatchSize": 1
    },
    "Style": {
      "BatchSize": 1
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
      "BatchSize": 1
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
  "Feature": {
    "Vehicle": {
      "Enable": true,
      "EnableDetection": true,
      "EnableType": true,
      "EnableColor": true,
      "EnablePlate": true,
      "EnablePlateEnhanced": false,
      "EnableMarker": true,
      "EnableFeatureVector": true
    },
    "Face": {
      "Enable": true,
      "EnableDetection": true,
      "EnableFeatureVector": true
    }
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