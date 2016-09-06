# Matrix Engine
### Version 0.5.2
2016-09-06

### Feature
 - DEEPVIDEO-576 Ranker支持多线程和高并发请求
 - DEEPVIDEO-583 车牌颜色新模型
 
### Bug
- DEEPVIDEO-291	Matrix - 与单机版数据服务集成
- DEEPVIDEO-322	ranker接口post请求时type先设成0再设成1，服务挂掉
- DEEPVIDEO-572	matrix threads 线程开多，启动过程CPU占用太多，影响整个系统性能
- DEEPVIDEO-575	matrix 相同threads，ranker在gpu_0_1上面申请显存不同，在gpu_1上面单独启动，coredump
- DEEPVIDEO-579	解决车牌识别时出现两个省份的问题
- DEEPVIDEO-588	matrix merge data_conf.json，805.txt的PATH错误，无法下载成功

