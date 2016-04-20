#include <gtest/gtest.h>
#include <ostream>
#include <fstream>

#include "codec/base64.h"
using namespace dg;
using namespace std;

void AssertStringEncode(string in, string out)
{
    char result[2*BUFSIZ];
    int len = Base64::Encode(in.c_str(), in.size(), result);
    result[len] = 0;
    EXPECT_EQ(out, result);
}

void AssertStreamEncode(string in, string out)
{
    stringstream s_in, s_out;
    s_in << in;
    Base64::Encode(s_in, s_out);
    EXPECT_EQ(out, s_out.str());
}

void AssertStringDecode(string in, string out)
{
    char result[BUFSIZ];
    int len = Base64::Decode(in.c_str(), in.size(), result);
    result[len] = 0;
    EXPECT_EQ(out, result);
}

void AssertStreamDecode(string in, string out)
{
    stringstream s_in, s_out;
    s_in << in;
    Base64::Decode(s_in, s_out);
    EXPECT_EQ(out, s_out.str());
}

TEST(Codec, Base64Encode)
{
    AssertStringEncode("a", "YQ==\n");
    AssertStringEncode("abcd", "YWJjZA==\n");

    AssertStreamEncode("a", "YQ==\n");
    AssertStreamEncode("abcd", "YWJjZA==\n");
}

TEST(Codec, Base64Decode)
{
    AssertStreamDecode("YQ==", "a");
    AssertStreamDecode("YWJjZA==", "abcd");

    AssertStreamDecode("YQ==", "a");
    AssertStreamDecode("YWJjZA==", "abcd");
}

TEST(Codec, Base64File)
{
    ifstream file("sample.jpg", ios::in|ios::binary|ios::ate);
    EXPECT_EQ(true, file.is_open());

    streampos size = file.tellg();
    char *buffer = new char[size];
    file.seekg(0, ios::beg);
    file.read(buffer, size);
    file.close();

    vector<char> input(buffer, buffer + size);
    string encoded = Base64::Encode(input);

    vector<char> output;
    Base64::Decode(encoded, output);


    EXPECT_EQ(input.size()+1, output.size());

    for(int i = 0; i < input.size() && i < output.size(); i ++)
    {
        EXPECT_EQ(input[i], output[i]);
    }
}



