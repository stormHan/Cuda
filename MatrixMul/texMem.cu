/*
	Use the texture memory to optimize the Matrix Multiplication

	2016.05.14

*/


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <Windows.h>
#include <time.h>

texture<float, 2, cudaReadModeElementType> texA;
texture<float, 2, cudaReadModeElementType> texB;

__global__ void MatrixMul(float *c, unsigned int w, unsigned int h)
{
	float sum = 0;
	//找出该线程所在的行和列
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	/*
	//计算纹理坐标
	float u = row / (float)w;
	float v = col / (float)h;

	//线程Thread(row, col)负责计算C(row, col)
	u -= 0.5f;
	v -= 0.5f;
	*/

	for (int i = 0; i < w; ++i)
	{
		sum += tex2D(texA, i, row) * tex2D(texB, col, i);
	}

	c[row * w + col] = sum;
}

__global__ void checkTex(float* C, float* D)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	C[row * blockDim.x * gridDim.x + col] = tex2D(texA, col, row);
	D[row * blockDim.x * gridDim.x + col] = tex2D(texB, col, row);
}

void MatrixMulCPU(float *_C, const float* _A, const float* _B, int WA, int HA, int WB, int HB)
{
	if (WA != HB)
	{
		printf("the matrix A and B cannot be multipled!");
		exit(0);
	}

	for (int i = 0; i < HA; ++i)
	{
		for (int j = 0; j < WB; ++j)
		{
			for (int k = 0; k < WA; ++k)
			{
				_C[i * WA + j] += _A[i * WA + k] * _B[k * WB + j];
			}
		}
	}
}
//给初始的矩阵一个随机值
void randomInit(float* _data, int _size)
{
	for (int i = 0; i < _size; ++i)
	{
		_data[i] = rand() / (float)RAND_MAX * 100;
	}
}

bool CheckAnswer(const float* _C, const float* _D, unsigned int size)
{
	bool isRight = true;
	for (int i = 0; i < size && isRight == true; ++i)
	{
		if (_C[i] != _D[i])
			isRight = false;
	}

	return isRight;
}

//print the matrix
void printMatrix(float* m_Matrix, int W, int H)
{
	for (int i = 0; i < W * H; ++i)
	{
		if (i % W == 0 && i != 0) printf("\n");
		printf("%2.1f ", m_Matrix[i]);
	}
	printf("\n\n");
}

int main()
{
	//用cuda事件来记录时间
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	const int width = 4;
	const int height = 4;

	float *B = (float *)malloc(sizeof(float) * height * width);
	float *A = (float *)malloc(sizeof(float) * height * width);
	float *C = (float *)malloc(sizeof(float) * height * width);
	float *D = (float *)malloc(sizeof(float) * height * width);

	memset(A, 0.0, sizeof(float) * height * width);
	memset(B, 0.0, sizeof(float) * height * width);
	memset(C, 0.0, sizeof(float) * height * width);
	memset(D, 0.0, sizeof(float) * height * width);

	srand((unsigned)time(0));

	randomInit(B, height * width);
	randomInit(A, height * width);

	//CPU version
	unsigned int tick1 = GetTickCount();
	MatrixMulCPU(D, A, B, width, height, width, height);
	printf("CPU use time : %dms\n", GetTickCount() - tick1);

	

	cudaChannelFormatDesc channelDescA = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDescB = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray* mat_A;
	cudaArray* mat_B;

	cudaMallocArray(&mat_A, &channelDescA, width, height);
	cudaMallocArray(&mat_B, &channelDescB, width, height);

	cudaMemcpyToArray(mat_A, 0, 0, A, sizeof(float) * height * width, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(mat_B, 0, 0, B, sizeof(float) * height * width, cudaMemcpyHostToDevice);

	//texA.addressMode[0] = cudaAddressModeWrap;
	//texA.addressMode[1] = cudaAddressModeWrap;
	texA.filterMode = cudaFilterModePoint;
	texA.normalized = false;
	//texB.addressMode[0] = cudaAddressModeWrap;
	//texB.addressMode[1] = cudaAddressModeWrap;
	texB.filterMode = cudaFilterModePoint;
	texB.normalized = false;
	

	cudaBindTextureToArray(texA, mat_A, channelDescA);
	cudaBindTextureToArray(texB, mat_B, channelDescB);
	
	

	float* d_C = NULL;
	//float* d_D = NULL;
	cudaMalloc(&d_C, width * height * sizeof(float));
	//cudaMalloc((void**)&d_D, width * height * sizeof(float));

	int block_size = 2;

	dim3 Threads(block_size, block_size);
	dim3 Blocks(width / block_size, height / block_size);

	MatrixMul << < Threads, Blocks >> >(d_C, width, height);
	//checkTex << < Threads, Blocks >> >(d_C, d_D);
	cudaMemcpy(C, d_C, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	//cudaMemcpy(D, d_D, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elaspsedTime;
	cudaEventElapsedTime(&elaspsedTime, start, stop);
	printf("GPU with Texutre Memory time : %3.1fms \n", elaspsedTime);

	if (!CheckAnswer(C, D, height * width))
		printf("The answer is wrong!\n");
	else printf("The answer is right!\n");

	printMatrix(A, 4, 4);
	printMatrix(B, 4, 4);
	printMatrix(C, 4, 4);
	printMatrix(D, 4, 4);


	cudaFree(d_C);
	cudaFree(mat_A);
	cudaFree(mat_B);
	
	return 0;
}