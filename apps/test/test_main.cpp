#include "gtest/gtest.h"
#include <iostream>
#include "services/image_service.h"

using namespace std;
using namespace dg;

TEST(FirstTest, test1) {
    ImageService imageService;
    EXPECT_TRUE(true);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
