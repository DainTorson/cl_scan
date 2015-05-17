#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

using namespace std;

#define BLOCK_SIZE 16
#define CHECK(err)                                                       \
    do{                                                                     \
        if (err != CL_SUCCESS){                                          \
            printf("ERROR code %d on line %d\n", err, __LINE__-1);       \
            return -1;                                                   \
        }                                                                \
    } while (0)



const char * loadKernel(char * filename)
{
	string kernel = "";
	ifstream file(filename);
	string line;
	while(getline(file, line))
	{
		kernel += line;
	}

	char * writable = new char[kernel.size() + 1];
	copy(kernel.begin(), kernel.end(), writable);
	writable[kernel.size()] = '\0';

	return writable;
}

float * loadData(char * filename, int &inputLength)
{
	ifstream file(filename);
	float * inputData;

	if(file.is_open())
	{
		file >> inputLength;
		inputData = (float * )malloc(inputLength*sizeof(float));

		for(int index = 0; index < inputLength && !file.eof(); ++index)
		{
			file >> inputData[index];
		}
	}

	return inputData;
}

template<class T> void printArray(T arr[], int size)
{
	for(int index = 0; index < size; ++index)
	{
		cout << arr[index] << " ";
	}

	cout << endl;
}

void scan(float * hostInput, float * hostOutput, int inputLength)
{
	hostOutput[0] = hostInput[0];

	for(int index = 1; index < inputLength; ++index)
	{
		hostOutput[index] = hostInput[index] + hostOutput[index - 1];
	}
}

int parallelScan(float * hostInput, float * hostOutput, int inputLength)
{
	cl_mem deviceInput;
	cl_mem deviceOutput;

	cl_int cl_error = CL_SUCCESS;
	cl_platform_id platform;
	cl_error = clGetPlatformIDs(1, &platform, NULL);
	CHECK(cl_error);
	
	cl_device_id device = (cl_device_id)malloc(sizeof(cl_device_id)*1000);
	cl_error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	CHECK(cl_error);
	
	cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &cl_error); 
	CHECK(cl_error);
	
	size_t param_size;
	cl_error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &param_size);
	CHECK(cl_error);
	
	cl_device_id * cl_devices = (cl_device_id*)malloc(param_size);
	cl_error = clGetContextInfo(context, CL_CONTEXT_DEVICES, param_size, cl_devices, NULL);
	CHECK(cl_error);
	
	cl_command_queue cl_cmd_queue = clCreateCommandQueue(context, cl_devices[0], NULL, &cl_error);
	CHECK(cl_error);
	const char * scansrc = loadKernel("kernel.c");

	cl_program cl_prgm;
	cl_prgm = clCreateProgramWithSource(context, 1, &scansrc, NULL, &cl_error);
	CHECK(cl_error);
	
	char cl_compile_flags[4096];
	sprintf(cl_compile_flags, "-cl-mad-enable");
	
	cl_error = clBuildProgram(cl_prgm, 0, NULL, cl_compile_flags, NULL, NULL);
	CHECK(cl_error);
	
	cl_kernel kernel = clCreateKernel(cl_prgm, "scan", &cl_error);
	CHECK(cl_error);

	deviceInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLength * sizeof(float),
								 hostInput, &cl_error);
	deviceOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inputLength * sizeof(float), NULL, NULL);
	CHECK(cl_error);

	size_t global_item_size = ((inputLength - 1) /2*BLOCK_SIZE + 1) * BLOCK_SIZE;
	size_t local_item_size = BLOCK_SIZE;

	int block_size = BLOCK_SIZE;
	cl_error = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &deviceInput);
	CHECK(cl_error);
	cl_error = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &deviceOutput);
	CHECK(cl_error);
	cl_error = clSetKernelArg(kernel, 2, sizeof(int), &inputLength);
	CHECK(cl_error);

	cl_event event = NULL;
	cl_error = clEnqueueNDRangeKernel(cl_cmd_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
	CHECK(cl_error);
	
	cl_error = clWaitForEvents(1, &event);
	CHECK(cl_error);

	clEnqueueReadBuffer(cl_cmd_queue, deviceOutput, CL_FALSE, 0, inputLength * sizeof(float), hostOutput, 0, NULL, NULL);

	clReleaseMemObject(deviceInput);
	clReleaseMemObject(deviceOutput);
	delete [] scansrc;
}

int main(void)
{
	float *hostInput;
	float *hostOutput;
	int inputLength;
	bool parallelExecution = true;

	hostInput = loadData("input_data.txt", inputLength);
	hostOutput = (float *)malloc(inputLength*sizeof(float));

	printArray(hostInput, inputLength);

	if(parallelExecution)
	{
		int check = parallelScan(hostInput, hostOutput, inputLength);
		if(check == -1)
		{
			system("PAUSE"); 
			return -1;
		}
	}
	else
	{
		scan(hostInput, hostOutput, inputLength);
	}
	
	printArray(hostOutput, inputLength);

	free(hostInput);
	free(hostOutput);
	system("PAUSE"); 
	return 0;
}