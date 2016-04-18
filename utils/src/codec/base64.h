/*============================================================================
 * File Name   : base64.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/
#ifndef MATRIX_UTIL_CODEC_BASE64_H_
#define MATRIX_UTIL_CODEC_BASE64_H_

#include <iostream>
#include <sstream>
#include <vector>

extern "C"
{
#include "cencode.h"
#include "cdecode.h"    
}

using namespace std;

namespace dg {
class Base64 {
public:
    static int Encode(char value)
    {
        return base64_encode_value(value);
    }

    static int Encode(const char* buffer_in, int buffer_in_len, char* buffer_out)
    {
        int buffer_out_len;

        base64_encodestate state;
        base64_init_encodestate(&state);

        buffer_out_len = base64_encode_block(buffer_in, buffer_in_len, buffer_out, &state);
        buffer_out_len += encodeEnd(buffer_out + buffer_out_len, &state);
        return buffer_out_len;
    }

    static void Encode(istream& input, ostream& output)
    {
        char buffer_out[2*BUFSIZ], buffer_in[BUFSIZ];
        int buffer_out_len, buffer_in_len;

        base64_encodestate state;
        base64_init_encodestate(&state);

        while(1)
        {
            input.read(buffer_in, BUFSIZ);
            buffer_in_len = input.gcount();
            if(input.fail() || buffer_in_len <= 0)
            {
                break;
            }

            buffer_out_len = base64_encode_block(buffer_in, buffer_in_len, buffer_out, &state);
            output.write((const char*)buffer_out, buffer_out_len);
        }

        buffer_out_len = encodeEnd(buffer_out, &state);
        output.write(buffer_out, buffer_out_len);
    }

    template<typename T>
    static string Encode(vector<T> input_data)
    {
        stringstream ss_in, ss_out;
        for(int i = 0; i < input_data.size(); i ++)
        {
            ss_in.write((char *)&input_data[i], sizeof(T));
        }
        Encode(ss_in, ss_out);
        return ss_out.str();
    }

    static int Decode(char value)
    {
        return base64_decode_value(value);
    }

    static int Decode(const char* buffer_in, int buffer_in_len, char* buffer_out)
    {
        base64_decodestate state;
        base64_init_decodestate(&state);

        return base64_decode_block(buffer_in, buffer_in_len, buffer_out, &state);
    }

    static void Decode(istream& input, ostream& output)
    {
        char buffer_out[BUFSIZ], buffer_in[BUFSIZ];
        int buffer_out_len, buffer_in_len;

        base64_decodestate state;
        base64_init_decodestate(&state);

        while(1)
        {
            input.read(buffer_in, BUFSIZ);
            buffer_in_len = input.gcount();
            if(input.fail() || buffer_in_len <= 0)
            {
                break;
            }

            buffer_out_len = base64_decode_block(buffer_in, buffer_in_len, buffer_out, &state);
            output.write((const char *)buffer_out, buffer_out_len);
        }
    }

    template<typename T>
    static void Decode(string base64_string, vector<T> &result_array)
    {
        stringstream ss_in, ss_out;
        ss_in << base64_string;
        Decode(ss_in, ss_out);

        T temp;
        while(ss_out.good())
        {
            ss_out.read((char *) (&temp), 1);
            result_array.push_back(temp);
        }
    }

private:
    static int encodeEnd(char* buffer_in, base64_encodestate *state)
    {
        return base64_encode_blockend(buffer_in, state);
    }
}; //end of class Base64
} //end of namespace dg

#endif // MATRIX_UTIL_CODEC_BASE64_H_