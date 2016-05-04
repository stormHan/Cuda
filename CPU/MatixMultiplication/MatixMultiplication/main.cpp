#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <string>

#define Width 100
#define Height 100

using namespace std;



int main()
{
	//fstream file("data.txt");
	
	int A[Width][Height];
	int B[Width][Height];
	int C[Width][Height];
	
	//srand((unsigned)time(NULL));
	for (int i = 0; i < Width; ++i)
	{
		for (int j = 0; j < Height; ++j)
		{
			A[i][j] = rand() % 100;
			B[i][j] = rand() % 100;
		}
	}
	int count = 0;
	for (int i = 0; i < Width; ++i)
	{
		for (int j = 0; j < Height; ++j)
		{
			C[i][j] = 0;
			for (int k = 0; k < Width; ++k)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
		}
		count++;
		printf("complete %d  / 1000.\n ", count);
	}

	cout << "finished!" << endl;
	return 0;
}