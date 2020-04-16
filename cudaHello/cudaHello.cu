#include <stdio.h>

__global__
void hello(int k) {
	printf("my thread number: %d %d\n", threadIdx.x, blockIdx.x);
	printf("Argument: %d\n", k);
}

int main() {
	
	hello<<<2,16>>>(5);
	cudaDeviceSynchronize();
}
