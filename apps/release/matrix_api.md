## Matrix Apps APIs


### Witness APIs

#### Image recognization api
gRPC:
```
    rpc Recognize (WitnessRequest) returns (WitnessResponse){}
```
HTTP:
```
    Method: POST
    Path: /rec/image
    Body: [json format of WitnessRequest]
    Return: [json format of WitnessResponse]
    Errors:
        200 "ok"
        400 "request msg invalid"
        500 "service error"
```
#### Batch images recognization api
gRPC:
```
    rpc BatchRecognize (WitnessBatchRequest) returns (WitnessBatchResponse) {}
```
HTTP:
```
    Method: POST
    Path: /rec/image/batch
    Body: [json format of WitnessBatchRequest]
    Return: [json format of WitnessBatchResponse]
    Errors:
         200 "ok"
         400 "request msg invalid"
         500 "service error"
```
#### Index api - id vs name
gRPC:
```
    rpc GetIndex (IndexRequest) returns (IndexResponse) {}
```
HTTP:
```
    Method: POST
    Path: /rec/index
    Body: [json format of IndexRequest]
    Return: [json format of IndexResponse]
    Errors:
         200 "ok"
         400 "request msg invalid"
         500 "service error"
```

### System APIs
#### Ping api
gRPC:
```
    rpc Ping(PingRequest) returns (PingResponse) {}
```
HTTP:
```
    Method: GET
    Path: /ping
    Return: [json format of PingResponse]
    Errors:
         200 "ok"
         400 "request msg invalid"
         500 "service error"

```
#### Get system status api
gRPC:
```
    rpc SystemStatus(SystemStatusRequest) returns (SystemStatusResponse) {}
```
HTTP:
```
    Method: GET
    Path: /info
    Return: [json format of SystemStatusResponse]
    Errors:
         200 "ok"
         400 "request msg invalid"
         500 "service error"
```

### Ranker APIs
#### Get ranked vector api
gRPC:
```
	rpc GetRankedVector(FeatureRankingRequest) returns (FeatureRankingResponse) {}
```
HTTP:
```
    Method: POST
    Path: /rank
    Body: [json format of FeatureRankingRequest]
    Return: [json format of FeatureRankingRequest]
    Errors:
         200 "ok"
         400 "request msg invalid"
         500 "service error"
```

```
message VideoMetadata {
    int64 Timestamp = 1;
    int32 Duration = 2;
    int32 SensorId = 3;
    string SensorName = 4;
    string SensorUrl = 5;
    int32 repoId =6;
    string repoInfo=7;
}

message Cutboard {
    uint32 X = 1;
    uint32 Y = 2;
    uint32 Width = 3;
    uint32 Height = 4;
    uint32 ResWidth = 5;
    uint32 ResHeight = 6;
    float Confidence = 7;
}
message CutboardImage {
    VideoMetadata Metadata = 1;
    Cutboard Cutboard = 2;
    Image Img = 3;
}
message Color {
    int32 ColorId = 1; //DEPLICATE!
    float Confidence = 2; //颜色汉字表示
    string ColorName = 3; //颜色识别置信度
}

enum ObjType {
    OBJ_TYPE_UNKNOWN = 0;
    OBJ_TYPE_CAR = 1;
    OBJ_TYPE_BICYCLE = 2;
    OBJ_TYPE_TRICYCLE = 3;
    OBJ_TYPE_PEDESTRIAN = 4;
    OBJ_TYPE_FACE = 1024;
    OBJ_TYPE_VEHICLE_VECTOR = 2048;
    OBJ_TYPE_BICYCLE_VECTOR = 2049;
    OBJ_TYPE_TRICYCLE_VECTOR = 2050;
    OBJ_TYPE_PEDESTRIAN_VECTOR = 2051;
    OBJ_TYPE_FACE_VECTOR = 3072;
    OBJ_TYPE_VEHICLE_CUTBOARD = 4096;
    OBJ_TYPE_BICYCLE_CUTBOARD = 4097;
    OBJ_TYPE_TRICYCLE_CUTBOARD = 4098;
    OBJ_TYPE_PEDESTRIAN_CUTBOARD = 4099;
    OBJ_TYPE_FACE_CUTBOARD = 5120;
}

enum DataFmtType {
    UNKNOWNFMT = 0;
    JSON = 1;
    PROTOBUF = 2;
    CSV = 3;
    PDF = 4;
}


message FeatureVector {
	int64 Id = 1; 		//特征的关键字
	string Feature = 2;	//特征值
}

message NullMessage{

}


message Image {
    string Id = 1;
    int32 Width = 2;
    int32 Height = 3;
    string URI = 4;
    string BinData = 5; // base64 format image
}

message RecFace {
    int64 Id = 1;
    CutboardImage Img = 2;
    bytes Features = 3;
    float Confidence = 4;
}

message RecVehicle {
    int64 Id = 1;
    VehicleModelType ModelType = 2;
    Color Color = 3;
    LicensePlate Plate = 4;
    CutboardImage Img = 5;
    repeated VehicleSymbol Symbols = 6;
    bytes Features = 7;
    ObjType VehicleType=8;
    string VehicleTypeName=9;
}

message VehicleModelType {
    int64 Id = 1; // business id
    int32 BrandId = 2;
    int32 SubBrandId = 3;
    int32 ModelYearId = 4;
    float Confidence = 5;
    string Type = 6; //车模类型名称
    string Brand = 7; //车模主品牌名称
    string SubBrand = 8; //车模子品牌名称
    string ModelYear = 9; //车模年款名称
    int32 TypeId = 10;
    string Model = 11;
    int32 IsHead = 12;
}
message LicensePlate {
    string PlateText = 1;
    Color Color = 2;
    int32 TypeId = 3;
    string TypeName = 4;
    float Confidence = 5;
    Cutboard Cutboard = 6;
}
message VehicleSymbol {
    int32 SymbolId = 1;
    string SymbolName = 2;
    repeated Symbol Symbols=3;
}
message Symbol{
    Cutboard cutboard =1;
    float Confidence=2;
}
enum DBType{
	KAFKA =0;
}
message StorageConfig {
	string Address = 1;         //结构化信息输出地址 ip:port
	DBType Type= 2;			//
	repeated string Tags = 3;   //设定数据的信息标签
}

enum RecognizeType {
    REC_TYPE_DEFAULT = 0;
    REC_TYPE_VEHICLE = 1;
    REC_TYPE_FACE = 2;
    REC_TYPE_ALL = 3;
}

enum RecognizeFunctions {
    RECFUNC_NONE = 0;
    RECFUNC_VEHICLE = 1;
    RECFUNC_VEHICLE_DETECT = 2;
    RECFUNC_VEHICLE_TRACK = 3;
    RECFUNC_VEHICLE_STYLE = 4;
    RECFUNC_VEHICLE_COLOR = 5;
    RECFUNC_VEHICLE_MARKER = 6;
    RECFUNC_VEHICLE_PLATE = 7;
    RECFUNC_VEHICLE_FEATURE_VECTOR = 8;
    RECFUNC_FACE = 9;
    RECFUNC_FACE_DETECTOR = 10;
    RECFUNC_FACE_FEATURE_VECTOR = 11;
}

message Time {
    int64 Seconds = 1;
    int64 NanoSecs = 2;
}
message MatrixError {
    int32 Code = 1;
    string Message = 2;
}
message PingRequest {

}

message PingResponse {
    string Message = 1;
}

message SystemStatusRequest {

}

message SystemStatusResponse {
    string ModelVer = 1;    //模型版本
    string EngineVer = 2;   //引擎版本
    string ServiceVer = 3;  //服务版本
    string CpuUsage = 4;    //CPU使用率
    string AvailMem = 5;    //可用内存
    string TotalMem = 6;    //全部内存
    string AvailDisk = 7;   //可用磁盘
    string TotalDisk = 8;   //全部磁盘
    string NetIOSend = 9;   //网络发送速度
    string NetIORecv = 10;  //网络接收速度
    string GpuUsage = 11;   //GPU使用率
    string GpuTotalMem = 12;//全部显存
}
enum IndexType {
    INDEX_DEFAULT = 0;
    INDEX_CAR_TYPE = 1;
    INDEX_CAR_MAIN_BRAND = 2;
    INDEX_CAR_SUB_BRAND = 3;
    INDEX_CAR_YEAR_MODEL = 4;
    INDEX_CAR_COLOR = 5;
    INDEX_CAR_MARKER = 6;
    INDEX_CAR_PLATE_TYPE = 7;
    INDEX_CAR_PLATE_COLOR = 8;
}

message IndexRequest {
    IndexType IndexType = 1;
}

message IndexResponse {
    map<int32, string> Index = 1;
}

message WitnessRequest {
    WitnessRequestContext Context = 1;
    WitnessImage Image = 2;
}

message WitnessBatchRequest {
    WitnessRequestContext Context = 1;
    repeated WitnessImage Images = 2;
}

message WitnessResponse {
    WitnessResponseContext Context = 1;
    WitnessResult Result = 2;
}

message WitnessBatchResponse {
    WitnessResponseContext Context = 1;
    repeated WitnessResult Results = 2;
}

message WitnessRelativeROI{
    int32 PosX=1;
    int32 PosY=2;
    int32 Width=3;
    int32 Height=4;
}
message WitnessMarginROI{
    int32 Left=1;
    int32 Top=2;
    int32 Right=3;
    int32 Bottom=4;
}

message WitnessRequestContext {
    string SessionId = 1; //请求Session
    string UserName = 2;     //用户名信息，需要开启验证功能
    string Token = 3;        //Token验证，需要开启验证功能
    repeated RecognizeFunctions Functions = 4; //功能列表
    RecognizeType Type = 5; //识别类型 enum RecognizeType
    StorageConfig Storage = 6; //Engine结果存储的信息
    map<string, string> Params = 7; //扩展参数

}

message WitnessImage {
    Image Data = 1;
    VideoMetadata WitnessMetaData =2;
    repeated WitnessRelativeROI RelativeRoi= 3;
    repeated WitnessMarginROI MarginRoi = 4;
}

message WitnessResponseContext {
    string SessionId = 1; //响应Session，如果请求Session不存在，生成UUID，否则复制请求的Session
    string Status = 2; //响应状态信息
    string Message = 3; //状态为错误的响应错误信息
    Time RequestTs = 4; //请求下发给Engine的时间
    Time ResponseTs = 5; //从Engine读取内容的时间
    map<string, Time> DebugTs = 6; //Debug用时
}

message WitnessResult {
    string InnerStatus = 1; //响应的内部状态
    string InnerMessage = 2; //响应的内部信息
    WitnessImage Image = 3; //响应的图片信息
    repeated RecVehicle Vehicles = 4; //识别的车辆列表
    repeated RecFace Faces = 5; //识别的人脸列表
    repeated RecPedestrian Pedestrians = 6; //识别的行人列表

}
message FeatureRankingRequest {
    int64 ReqId = 1;		//请求的Id
    RecognizeType Type = 2; //
    Image Image = 3; 		//待比对图片信息
    repeated Cutboard InterestedAreas = 4;//比对设置的感兴趣区域
    repeated Cutboard DisabledAreas = 5;//比对设置的禁用区域（当前不可用）
    repeated FeatureVector Candidates = 6;//待比对的特征列表
    int32 Limit = 7;		//比对输出条目限制（0为不限制）
}

message FeatureRankingResponse {
    int64 ReqId = 1;			//比对返回的请求Id(对应请求的ReqId)
    repeated int64 Ids = 2;		//比对返回的排序结果
    repeated float Scores = 3;	//比对返回的相关性分值
}
```