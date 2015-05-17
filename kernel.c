__kernel void scan(__global float * input, __global float * output, int length)
{

	int block_size = get_local_size(0);
	__local float ds_array[32];

	int gid = get_global_id(0);
	int lid = get_local_id(0);

	if(gid < length)
	{
		ds_array[lid] = input[gid];
	}
	else
	{
		ds_array[lid] = 0;
	}

	if(gid + block_size < length)
	{
		ds_array[lid + block_size] = input[gid + block_size];
	}
	else
	{
		ds_array[lid + block_size] = 0;
	}

	for(int stride = 1; stride <= block_size; stride *= 2)
	{
		int index = (lid + 1)*stride*2 - 1;
		
		if(index < 2*block_size)
			ds_array[index] += ds_array[index - stride];
		barrier(CLK_LOCAL_MEM_FENCE);	
	}
	
	for(int stride = block_size/2; stride > 0; stride /= 2)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		int index = (lid + 1)*stride*2 - 1;
		
		if(index + stride < 2*block_size)
			ds_array[index + stride] += ds_array[index];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(gid < length)
	{
		output[gid] = ds_array[lid];
	}
	
	if(gid + block_size < length)
	{
		output[gid + block_size] = ds_array[lid];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
}