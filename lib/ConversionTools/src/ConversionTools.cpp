#include "ConversionTools.h"

void ConversionTools::uint8_to_int8(const uint8_t *src, int8_t *dst, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        dst[i] = src[i] - 128;
    }
}
void ConversionTools::int8_to_uint8(const int8_t *src, uint8_t *dst, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        dst[i] = src[i] + 128;
    }
}