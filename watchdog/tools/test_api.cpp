/*
 * test_api.cpp
 *
 *  Created on: May 9, 2016
 *      Author: jiajiachen
 */

#include <sys/file.h>


#include "gtest/gtest.h"
#include "watch_dog.h"
#include "dog.h"
//
//TEST(CheckFeatureTest,HandlesDogOnFeatureInput){
//    EXPECT_EQ(ERR_FEATURE_ON,CheckFeature(FEATURE_CAR_DETECTION,true));
//    EXPECT_EQ(ERR_FEATURE_ON,CheckFeature(FEATURE_CAR_DETECTION,false));
//
//}
//TEST(CheckFeatureTest,HandlesDogOffFeatureInput){
//    EXPECT_EQ(ERR_FEATURE_ON,CheckFeature(FEATURE_CAR_DETECTION,true));
//}
TEST(CurrPerformanceTest,HandlesFeatureInput){
    resetDogData(0);

    unsigned long long pef;
    ASSERT_EQ(KEY_OK,SetMaxPerformance(FEATURE_CAR_DETECTION,30));
    GetMaxPerformance(FEATURE_CAR_DETECTION,pef);
    EXPECT_EQ(30,pef);
    ASSERT_EQ(KEY_OK,SetCurrPerformance(FEATURE_CAR_DETECTION,10000));
    GetCurrPerformance(FEATURE_CAR_DETECTION,pef);
    EXPECT_EQ(1,pef);
    ASSERT_EQ(KEY_OK,SetCurrPerformance(FEATURE_CAR_DETECTION,10000));
    GetCurrPerformance(FEATURE_CAR_DETECTION,pef);
    EXPECT_EQ(2,pef);
}

TEST(GetDogTimeTest,HandlesNoInput){
    time_t time;
    GetDogTime(time);
    printf("%d\n",time);

}

TEST(MaxPerformanceTest,HandlesFeatureInput){

    unsigned long long pef;

    ASSERT_EQ(KEY_OK,SetMaxPerformance(FEATURE_CAR_DETECTION,30));
    GetMaxPerformance(FEATURE_CAR_DETECTION,pef);
    EXPECT_EQ(30,pef);

    ASSERT_EQ(KEY_OK,SetMaxPerformance(FEATURE_CAR_STYLE,40));
    GetMaxPerformance(FEATURE_CAR_STYLE,pef);
    EXPECT_EQ(40,pef);

    ASSERT_EQ(KEY_OK,SetMaxPerformance(FEATURE_CAR_COLOR,50));
    GetMaxPerformance(FEATURE_CAR_COLOR,pef);
    EXPECT_EQ(50,pef);

    ASSERT_EQ(KEY_OK,SetMaxPerformance(FEATURE_CAR_PLATE,60));
    GetMaxPerformance(FEATURE_CAR_PLATE,pef);
    EXPECT_EQ(60,pef);

    ASSERT_EQ(KEY_OK,SetMaxPerformance(FEATURE_CAR_MARK,70));
    GetMaxPerformance(FEATURE_CAR_MARK,pef);
    EXPECT_EQ(70,pef);

    ASSERT_EQ(KEY_OK,SetMaxPerformance(FEATURE_CAR_RANK,80));
    GetMaxPerformance(FEATURE_CAR_RANK,pef);
    EXPECT_EQ(80,pef);

    ASSERT_EQ(KEY_OK,SetMaxPerformance(FEATURE_CAR_EXTRACT,90));
    GetMaxPerformance(FEATURE_CAR_EXTRACT,pef);
    EXPECT_EQ(90,pef);

    ASSERT_EQ(KEY_OK,SetMaxPerformance(FEATURE_FACE_EXTRACT,100));
    GetMaxPerformance(FEATURE_FACE_EXTRACT,pef);
    EXPECT_EQ(100,pef);

    ASSERT_EQ(KEY_OK,SetMaxPerformance(FEATURE_FACE_RANK,110));
    GetMaxPerformance(FEATURE_FACE_RANK,pef);
    EXPECT_EQ(110,pef);
}
int main(int argc,char **argv){
//    FILE *fp = NULL;
//    if((fp=fopen("file_lock.test","w+"))==NULL)
//        printf("file open error!\n");
//    int i=flock(fileno(fp),LOCK_EX);
//    if(i==0){
//        printf("file opdffs %d!\n",i );
//        sleep(20);
//        flock(fileno(fp),LOCK_UN);
//    }
//
//
//    fclose(fp);
//    return 0;

    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
