#include "ImageFormater.h"
#include <stdio.h>
#include <string.h>
// #define DEBUGVIAOPENCV

#ifdef DEBUGVIAOPENCV
#include <opencv2/opencv.hpp>
#endif

#define PIXELMAX (127)
#define PIXELMIN (-128)
// images are inverse thresholded based on the darkest pixel plus this value
#define THRESHOLDNOISEOFFSET (50)
#define THRESHOLD (0)
void ImageFormater::SetElement(int8_t *img, uint width, uint row, uint col, int8_t value)
{
	img[width * row + col] = value;
}
int8_t ImageFormater::GetElement(int8_t *img, uint width, uint row, uint col)
{
	return img[width * row + col];
}
uint ImageFormater::GetCenterOfMassX(int8_t *img, uint width, uint height, int8_t threshold)
{
	uint center = 0;
	uint count = 0;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			if (this->GetElement(img, width, j, i) > threshold)
			{
				center += i;
				count++;
			}
		}
	}
	return center / count;
}
uint ImageFormater::GetCenterOfMassY(int8_t *img, uint width, uint height, int8_t threshold)
{
	uint center = 0;
	uint count = 0;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			if (this->GetElement(img, width, j, i) > threshold)
			{
				center += j;
				count++;
			}
		}
	}
	return center / count;
}
/// <summary>
/// this applys an inverting threshold
/// </summary>
/// <param name="thresLevel">pixels lower than this value are set to Max Pixels higher than this value are set to min</param>
/// <param name="img"></param>
/// <param name="length"></param>
/// <returns></returns>
void ImageFormater::applyThreshold(int8_t thresLevel, int8_t *img, uint length)
{
	std::cout << "Thres level " << (int)thresLevel << "\n";
	for (int i = 0; i < length; i++)
	{
		if (img[i] > thresLevel)
		{
			img[i] = PIXELMIN;
		}
		else
		{
			img[i] = PIXELMAX;
		}
	}
}
/// <summary>
/// thesholds relative to the darkets spot in the image
/// </summary>
/// <param name="img"></param>
/// <param name="length"></param>
/// <returns></returns>
void ImageFormater::applyDynamicThreshold(int8_t *img, uint length)
{
	// assume the darkest pixels are part of the digit
	int8_t min = PIXELMAX;
	for (int i = 0; i < length; i++)
	{
		if (img[i] < min)
			min = img[i];
	}
	applyThreshold(min + THRESHOLDNOISEOFFSET, img, length);
}

/// <summary>
/// This might be bads to do the orginal mnist dataset prepocessing didnt seam to have this step
/// modifies an image bymaking pixels adjacent to 255 valued pixels white
/// </summary>
/// <param name="img"> imag array</param>
/// <param name="width">width of image</param>
/// <param name="height">height of image</param>
void ImageFormater::applyThicken(int8_t *img, uint width, uint height)
{
	// make adjaent pixels also white
	int8_t *copy = new int8_t[width * height];

	// TODO this might work better if the value of a pixel is just transformed to be the average of its neighbors and self
	// that seems more cositanst with hwo minst was created
	for (int i = 0; i < width * height; i++)
	{
		copy[i] = PIXELMIN;
	}
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{

			int sum = GetElement(img, width, j, i) + 128;
			for (int t = -1; t < 2; t++)
			{
				for (int r = -1; r < 2; r++)
				{
					int x = i + t;
					int y = j + r;
					if (y > 0 && y < height)
					{
						if (x > 0 && x < width)
						{
							sum += GetElement(img, width, y, x) + 128;
						}
					}
				}
			}

			int t = ((sum / 9) - 128); // get average
			// increase brightness of non black pixels
			t = t == PIXELMIN ? t : t + 128;
			// Pixels may have passed Max Fix it if so
			t = t <= PIXELMAX ? t : PIXELMAX;
			SetElement(copy, width, j, i, t);
		}
	}
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			int value = GetElement(img, width, j, i);
			if (value == PIXELMAX)
			{
				SetElement(copy, width, j, i, value);
			}
		}
	}
	memcpy(img, copy, width * height);
	delete[] copy;
}
/// <summary>
/// takes a large image and abounding box of a potential digit and out puts a 20 by 20 digit that is scaled to fill toa n output image size
/// assume square
/// </summary>
/// <param name="img">larger image</param>
/// <param name="width">width of larger image</param>
/// <param name="height">height of larger iamge</param>
/// <param name="bleft">bounding box</param>
/// <param name="btop">bounding box</param>
/// <param name="bright">bounding box</param>
/// <param name="bbottom">bounding box</param>
/// <param name="outImg">Output image array</param>
/// <param name="outwidth">output image width</param>
/// <param name="outheight">output image height</param>
void ImageFormater::scale20by20digit(int8_t *img, uint width, uint height, uint bleft, uint btop, uint bright, uint bbottom, int8_t *outImg, uint outwidth, uint outheight)
{
	// TODO Un hardcode 20 to be out size
	// scale relative to largest axis to be 20 on the largest axis
	float centerRatioX = ((((float)(bright - bleft)) / 2.0) / ((float)(bright - bleft)));
	float centerRatioY = ((((float)(bbottom - btop)) / 2.0) / ((float)(bbottom - btop)));
	if (bright - bleft != bbottom - btop)
	{
		throw; // assume square
	}

	for (float i = 0; i < 20; i++)
	{
		for (float j = 0; j < 20; j++)
		{
			int x = bleft + (int)(i * ((float)(bright - bleft + 1)) / 20.0);
			int y = btop + (int)(j * ((float)(bbottom - btop + 1)) / 20.0);
			int value = GetElement(img, width, y, x);
			SetElement(outImg, 20, j, i, value);
		}
	}
#ifdef DEBUGVIAOPENCV
	cv::Mat img2 = cv::Mat(outwidth, outheight, CV_8S, outImg);
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	imshow("image", img2);
	cv::waitKey(0);
#endif
}
/// <summary>
/// create a minist image from a 20 by 20 digit image
/// </summary>
/// <param name="img20by20"> a 20 by 20 digit</param>
/// <param name="imgOut"> mnist image 28 by 28</param>
void ImageFormater::createMnistImageFromDigit(int8_t *img20by20, int8_t *imgOut)
{
	for (int i = 0; i < 28 * 28; i++)
	{
		imgOut[i] = PIXELMIN;
	}
	for (int i = 0; i < 20; i++)
	{
		for (int j = 0; j < 20; j++)
		{
			int value = GetElement(img20by20, 20, i, j);
			SetElement(imgOut, 28, i + 4, j + 4, value);
		}
	}
}
/// <summary>
/// calulates the radius of the smallest square that is centered around the center of mass
/// that contains the numbers
/// Note fails if there are gaps in the numbers.
/// </summary>
/// <param name="img"></param>
/// <param name="width"></param>
/// <param name="height"></param>
/// <param name="centerOfMassX"></param>
/// <param name="centerOfMassY"></param>
/// <returns></returns>
uint ImageFormater::GetBoundingBoxRadius(int8_t *img, uint width, uint height, uint centerOfMassX, uint centerOfMassY)
{
	uint left = 0;
	uint right = width - 1;
	uint top = 0;
	uint bottom = height - 1;
	int radius = 1;
	bool radiusPastShape = false;

	while (!radiusPastShape)
	{
		radiusPastShape = true;

		for (int i = 0; i < width; i++)
		{

			uint x = i;
			uint y1 = centerOfMassY - radius;
			uint y2 = centerOfMassY + radius;
			int8_t pixel1Value = PIXELMIN;
			int8_t pixel2Value = PIXELMIN;
			if (y1 > 0 && y1 < height - 1)
			{
				pixel1Value = GetElement(img, width, y1, x);
			}
			if (y2 > 0 && y2 < height - 1)
			{
				pixel2Value = GetElement(img, width, y2, x);
			}
			bool partOfNum = pixel1Value != PIXELMIN || pixel2Value != PIXELMIN;
			if (partOfNum)
			{
				radiusPastShape = false;
				break;
			}
		}
		for (int j = 0; j < height; j++)
		{
			uint x1 = centerOfMassY - radius;
			uint x2 = centerOfMassY + radius;
			uint y = j;
			int8_t pixel1Value = PIXELMIN;
			int8_t pixel2Value = PIXELMIN;
			if (x1 > 0 && x1 < width - 1)
			{
				pixel1Value = GetElement(img, width, y, x1);
			}
			if (x2 > 0 && x2 < width - 1)
			{
				pixel2Value = GetElement(img, width, y, x2);
			}
			bool partOfNum = pixel1Value != PIXELMIN || pixel2Value != PIXELMIN;
			if (partOfNum)
			{
				radiusPastShape = false;
				break;
			}
		}
		radius++;
	}
	return radius;
}

/// <summary>
/// take a larger picture and find a potential digit and crop it to be a 20 by 20 image of only the digit
/// </summary>
/// <param name="img">larger image</param>
/// <param name="width"> size of larger image</param>
/// <param name="height">size of larger aimge</param>
/// <param name="outImg"> 20 by 20 image</param>
void ImageFormater::crop20by20Digit(int8_t *img, uint width, uint height, int8_t *outImg)
{
	// find the digits bounding box
	uint8_t min = PIXELMAX;

	uint centerOfMassX = GetCenterOfMassX(img, width, height, THRESHOLD);
	uint centerOfMassY = GetCenterOfMassY(img, width, height, THRESHOLD);
	std::cout << "center X " << centerOfMassX;
	std::cout << "center Y " << centerOfMassY;
#ifdef DEBUGVIAOPENCV

	SetElement(img, 160, centerOfMassY, centerOfMassX, 0);
	SetElement(img, 160, 0, 0, 0);
	SetElement(img, 160, 0, 119, 0);
	SetElement(img, 160, 159, 0, 0);
	SetElement(img, 160, 159, 119, 0);
#endif
	int radius = GetBoundingBoxRadius(img, width, height, centerOfMassX, centerOfMassY);
	std::cout << "Radius " << radius;
#ifdef DEBUGVIAOPENCV

	SetElement(img, 160, centerOfMassY - radius, centerOfMassX - radius, 0);
	SetElement(img, 160, centerOfMassY - radius, centerOfMassX + radius, 0);
	SetElement(img, 160, centerOfMassY + radius, centerOfMassX - radius, 0);
	SetElement(img, 160, centerOfMassY + radius, centerOfMassX + radius, 0);
	cv::Mat img2 = cv::Mat(120, 160, CV_8S, img);
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	imshow("image", img2);
	cv::waitKey(0);
#endif
	// we have to take the area the digit is in and scale it to fit in a 20 by 20 window
	uint left = centerOfMassX - radius;
	uint top = centerOfMassY - radius;
	uint right = centerOfMassX + radius;
	uint bottom = centerOfMassY + radius;
	scale20by20digit(img, width, height, left, top, right, bottom, outImg, 20, 20);
}

/// <summary>
/// transforms an input image to an image that looks like a minst image (28*28  black and white image with digit in centered 20x20 area)
/// </summary>
void ImageFormater::CreateMnistImageFromImage(int8_t *inputImage, uint width, uint height, int8_t *mnistFormatedOutputimage)
{
	applyDynamicThreshold(inputImage, width * height);

	applyThicken(inputImage, width, height);
	applyThicken(inputImage, width, height);
	int8_t img20by20[20 * 20];

	crop20by20Digit(inputImage, width, height, img20by20);
	createMnistImageFromDigit(img20by20, mnistFormatedOutputimage);
}
