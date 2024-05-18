#include <iostream>
#include <iostream>
#pragma once
typedef unsigned int uint;
class ImageFormater
{
private:
	void SetElement(int8_t* img, uint width, uint row, uint col, int8_t value);
	int8_t GetElement(int8_t* img, uint width, uint row, uint col);
	uint GetCenterOfMassX(int8_t* img, uint width, uint height, int8_t threshold);
	uint GetCenterOfMassY(int8_t* img, uint width, uint height, int8_t threshold);
	uint GetBoundingBoxRadius(int8_t* img, uint width, uint height, uint centerOfMassX,uint centerOfMassY);
	void applyThreshold(int8_t thresLevel, int8_t* img, uint length);
	void applyDynamicThreshold(int8_t* img, uint length);
	void applyThicken(int8_t* img, uint width, uint height);
	void scale20by20digit(int8_t* img, uint width, uint height, uint bleft, uint btop, uint bright, uint bbottom, int8_t* outImg, uint outwidth, uint outheight);
	void createMnistImageFromDigit(int8_t* img20by20, int8_t* imgOut);
	void crop20by20Digit(int8_t* img, uint width, uint height, int8_t* outImg);

public:
	void CreateMnistImageFromImage(int8_t* inputImage, uint width, uint height, int8_t* mnistFormatedOutputimage);
};

