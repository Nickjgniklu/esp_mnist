#include <iostream>
#pragma once
typedef unsigned int uint;
class ConversionTools
{
public:
    static void uint8_to_int8(const uint8_t *src, int8_t *dst, size_t len);
    static void int8_to_uint8(const int8_t *src, uint8_t *dst, size_t len);
};
