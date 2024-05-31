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
void ConversionTools::uint8_rgb_to_uint8_grayscale(const uint8_t *rgb_src, uint8_t *gray_scale_dts, size_t grayscale_length)
{
    for (size_t i = 0; i < grayscale_length; i++)
    {
        gray_scale_dts[i] = (rgb_src[i * 3] + rgb_src[i * 3 + 1] + rgb_src[i * 3 + 2]) / 3;
    }
}
