#include "sys/time.h"

#include <stdio.h>

double getTimeStamp() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

// device-side matrix addition
/*__global__ void f_addmat(float* A, float* B, float* C, int nx, int ny) {
	// kernel code might look something like this
	// but you may want to pad the matrices and index into them accordingly
	int ix = threadId.x + bloackId.x * blockDim.x;
	int iy = threadId.y + bloackId.y * blockDim.y;
	int idx = iy * ny + ix;
	if ((ix < nx) && (iy < ny))
		C[idx] = A[idx] + B[idx];
}*/


__global__ void f_addmat(float* A, float* B, float* C, int nx, int ny) {
	// kernel code might look something like this
	// but you may want to pad the matrices and index into them accordingly
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	//int numElem=nx*ny;
	long long idx = iy * nx + ix;
	if ((ix < nx) && (iy < ny)){

		C[idx] = A[idx] + B[idx];

	}
}

void matrixSumHost(float* A, float* B, float* C, int nx, int ny)
{
	float* ia = A, * ib = B, * ic = C;
	for (int iy = 0; iy < ny; iy++) {
		for (int ix = 0; ix < nx; ix++)
			ic[ix] = ia[ix] + ib[ix];
		ia += nx; ib += nx; ic += nx;
	}
}

void initDataA(float *h_A, int nx, int ny) {
	int xcoord = 0;
	int ycoord = 0;
	for (int i = 0; i < ny*nx; i++) {
		h_A[i]= (float)(xcoord + ycoord) / 3.0;
		if (xcoord == nx - 1) {
			xcoord = 0;
			ycoord += 1;
		}
		else {
			xcoord += 1;
		}
		
	}
}

void initDataB(float* h_B, int nx, int ny) {
	int xcoord = 0;
	int ycoord = 0;
	for (int i = 0; i < ny * nx; i++) {
		h_B[i] = (float)(xcoord + ycoord)* 3.14;
		if (xcoord == nx - 1) {
			xcoord = 0;
			ycoord += 1;
		}
		else {
			xcoord += 1;
		}

	}
}
//int argc, char* argv[]
int main(int argc, char* argv[]) {

	if (argc != 3) {
		printf("Error: wrong number of args\n");
		//exit();
	}
	int nx = atoi(argv[1]); // should check validity
	int ny = atoi(argv[2]); // should check validity
	int noElems = nx * ny;
	int bytes = noElems * sizeof(float);
	//printf("my name");
	float* h_A = (float*)malloc(bytes);
	float* h_B = (float*)malloc(bytes);
	float* h_hC = (float*)malloc(bytes); // host result
	initDataA(h_A, nx,ny);
	initDataB(h_B, nx, ny);
	matrixSumHost(h_A, h_B, h_hC, nx, ny);
/* device side*/
	float* d_A, * d_B, * d_C;
	cudaMalloc((void**)& d_A, bytes);
	cudaMalloc((void**)& d_B, bytes);
	cudaMalloc((void**)& d_C, bytes);
	double timeStampA = getTimeStamp();

	float* h_dC = (float*)malloc(bytes);

	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
	double timeStampB = getTimeStamp();
	
	int blockx=32;
	int blocky=32;
	int marker=0;
	while (nx>blockx*65535){
		marker=1;
		blockx=2*blockx;
	}
	while (ny>blocky*65535){
		marker=2;
		blocky=2*blocky;
	}
	if (marker==1){
		blocky=1024/blockx;
	}
	dim3 block(blockx, blocky); // you will want to configure this

	int gridSizeX=(nx + block.x - 1) / block.x;
	if (gridSizeX>=65535){

		gridSizeX=65535;

	}

	int gridSizeY=(ny + block.y - 1) / block.y;
	if (gridSizeY>=65535){

		gridSizeY=65535;

	}
	dim3 grid(gridSizeX, gridSizeY);

	f_addmat <<<grid, block >>> (d_A, d_B, d_C, nx, ny);
	cudaDeviceSynchronize();
	double timeStampC = getTimeStamp();

	cudaMemcpy(h_dC, d_C, bytes, cudaMemcpyDeviceToHost);

	double timeStampD = getTimeStamp();
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	cudaDeviceReset();
	/*for (int j=0;j<10;j++){
		printf("%f %f\n",h_dC[j],h_hC[j]);
	}*/
	
	for (int i = 0; i < noElems; i++) {
		if (h_dC[i]!=h_hC[i]){

			printf("wrong %d\n",i);
			for (int j=i;j<i+10;j++){
				printf("%f %f\n",h_dC[j],h_hC[j]);
			}
			break;
		}
		
	}
	printf("total_time:%4f",timeStampD-timeStampA);
	printf("CPU_GPU_time:%4f",timeStampB-timeStampA);
	printf("kernel_time:%4f",timeStampC-timeStampB);
	printf("GPU_CPU_time:%4f",timeStampD-timeStampC);
	






//...............................................................................

	/*for (int i = 0; i < noElems; i++) {
		cout << h_A[i];
	}
	cout << " " << endl;
	for (int i = 0; i < noElems; i++) {
		cout << h_B[i];
	}
	cout << " " << endl;
	
	matrixSumHost(h_A, h_B, h_hC, nx, ny);
	for (int i = 0; i < noElems; i++) {
		cout << h_hC[i];
	}
	cout << " " << endl;*/

	//cout << h_A[900]<<endl;
/*
// get program arguments
	if (argc != 3) {
		printf("Error: wrong number of args\n");
		//exit();
	}
	int nx = atoi(argv[2]); // should check validity
	int ny = atoi(argv[3]); // should check validity
	int noElems = nx * ny;
	int bytes = noElems * sizeof(float);
	// but you may want to pad the matrices¡­
	// alloc memory host-side
	float* h_A = (float*)malloc(bytes);
	float* h_B = (float*)malloc(bytes);
	float* h_hC = (float*)malloc(bytes); // host result
	float* h_dC = (float*)malloc(bytes); // gpu result

	// init matrices with random data


	//initData(h_A, noElems); initData(h_B, noElems);


	// alloc memory dev-side
	float* d_A, * d_B, * d_C;
	cudaMalloc((void**)& d_A, bytes);
	cudaMalloc((void**)& d_B, bytes);
	cudaMalloc((void**)& d_C, bytes);

	double timeStampA = getTimeStamp();

	cudaMemcpy(d_A, h_A, bytes, cudaMemCpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemCpyHostToDevice);

	// note that the transfers would be twice as fast if h_A and h_B
	// matrices are pinned

	double timeStampB = getTimeStamp();

	// invoke Kernel
	dim3 block(32, 32); // you will want to configure this
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	f_addmat << <grid, block >> > (d_A, d_B, d_C, nx, ny);
	cudaDeviceSynchronize();

	double timeStampC = getTimeStamp();
	//copy data back
	cudaMemCpy(h_dC, d_C, bytes, cudaMemCpyDeviceToHost);
	double timeStampD = getTimeStamp();

	// free GPU resources
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	cudaDeviceReset();

	// check result
	h_addmat(h_A, h_B, h_hC, nx, ny);
	h_dC == h+hC???
	// print out results

*/

	
}
