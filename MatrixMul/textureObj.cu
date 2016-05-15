/*
	Practise the Texture memory to help me understand the cuda

	DateTime 2016.05.10

*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>
#include <Windows.h>

//Texture Object
//struct cudaTextureDesc
//{
//	enum cudaTextureAddressMode addressMode[3];
//	enum cudaTextureFilterMode	filterMode;
//	enum cudaTextureReadMode	readMode;
//
//	int							sRGB;
//	int							normalizedCoords;
//	unsigned int				maxAnisotropy;
//	enum cudaTextureFilterMode	mipmapFilterMode;
//	float						minMipmapLevelClamp;
//	float						maxMipmapLevelClamp;
//};

int main()
{
	const int width = 1024;
	const int height = 1024;

	//先申请CPU上的输入输出矩阵
	float *B = (float *)malloc(sizeof(float) * height * width);
	float *A = (float *)malloc(sizeof(float) * height * width);

	memset(A, 0.0, sizeof(float) * height * width);
	memset(B, 0.0, sizeof(float) * height * width);
	//开辟显存
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height); //分配cuArray
	
	//从host data区中src指向的存储区域拷贝数据到cudaArray里
	cudaMemcpyToArray(cuArray, 0, 0, A, sizeof(float) * 1024 * 1024, cudaMemcpyHostToDevice);

	//指定纹理
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	//指定纹理object的参数
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]	=	cudaAddressModeWrap;
	texDesc.addressMode[1]	=	cudaAddressModeWrap;
	texDesc.filterMode		=	cudaFilterModeLinear;
	texDesc.readMode		=	cudaReadModeElementType;
	texDesc.normalizedCoords =	1;

	//Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	//Allocate the output maxtrix
	float* C;
	cudaMalloc(&C, width * height * sizeof(float));

	//Invoke kernel
	dim3 dimBlock;
	dim3 dimGrid;


	//Destory texture object
	cudaDestroyTextureObject(texObj);

	//Free the device memory;
	cudaFreeArray(cuArray);
	cudaFree(C);

	free(A);
	free(B);

    return 0;
}
