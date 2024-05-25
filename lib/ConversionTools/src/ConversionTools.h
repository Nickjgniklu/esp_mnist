#include <iostream>
#pragma once
typedef unsigned int uint;
class ConversionTools
{
public:
    static void uint8_to_int8(const uint8_t *src, int8_t *dst, size_t len);
    static void int8_to_uint8(const int8_t *src, uint8_t *dst, size_t len);
    /// @brief Converts a RGB image to a grayscale image
    /// @param rgb_src Pointer to the RGB image
    /// @param gray_scale_dts Pointer to the grayscale image
    /// @param grayscale_length Length of the grayscale image
    /// @note The RGB image is expected to be 3 times the length of the grayscale image
    static void uint8_rgb_to_uint8_grayscale(const uint8_t *rgb_src, uint8_t *gray_scale_dts, size_t grayscale_length);
};
