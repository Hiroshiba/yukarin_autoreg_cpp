#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <chrono>

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand_kernel.h>
#include <cub.cuh>
#include <util_type.cuh>
#include <device_launch_parameters.h>

#include "CppWaveRNN.h"


#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif


template<typename T>
T* cudaMallocUtil(int size, T* h = NULL) {
	T* x;
	cudaErrorCheckUtil(cudaMalloc(&x, size * sizeof(T)));
	if (h != NULL) {
		cudaErrorCheckUtil(cudaMemcpy(x, h, size * sizeof(T), cudaMemcpyHostToDevice));
	}
	return x;
}


template<typename T>
struct ndarray {
	T* device;
	T* host;
	int shape1 = 1;
	int shape2 = 1;
	int shape3 = 1;
	ndarray(int s1, T* h = NULL) : shape1(s1), host(h) {
		device = cudaMallocUtil<T>(size(), host);
	}
	ndarray(int s1, int s2, T* h = NULL) : shape1(s1), shape2(s2), host(h) {
		device = cudaMallocUtil<T>(size(), host);
	}
	ndarray(int s1, int s2, int s3, T* h = NULL) : shape1(s1), shape2(s2), shape3(s3), host(h) {
		device = cudaMallocUtil<T>(size(), host);
	}

	int size() {
		return shape1 * shape2 * shape3;
	}
};


__global__ void concat(float* xl, int* x, float* l, float* x_embedder_W, int batch_size, int local_size, int embedding_size)
{
	int feature_size = embedding_size + local_size;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= batch_size * feature_size) { return; }

	int i_batch = i / feature_size;
	int i_feature = i % feature_size;

	if (i_feature < embedding_size) {
		// embedding
		int i_x = x[i_batch];
		int i_embedding = i_x * embedding_size + i_feature;
		xl[i] = x_embedder_W[i_embedding];
	}
	else {
		// local
		int i_local = i_batch * local_size + (i_feature - embedding_size);
		xl[i] = l[i_local];
	}
}

__global__ void gruElementWise(
	float* hidden,
	float* W,
	float* U,
	int batch_size,
	int hidden_size
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= batch_size * hidden_size) return;

	float W_r_x = W[i % hidden_size + hidden_size * 0 + hidden_size * 3 * (i / hidden_size)];
	float U_r_h = U[i % hidden_size + hidden_size * 0 + hidden_size * 3 * (i / hidden_size)];
	float r = tanh((W_r_x + U_r_h) * 0.5f) * 0.5f + 0.5f;

	float W_z_x = W[i % hidden_size + hidden_size * 1 + hidden_size * 3 * (i / hidden_size)];
	float U_z_h = U[i % hidden_size + hidden_size * 1 + hidden_size * 3 * (i / hidden_size)];
	float z = tanh((W_z_x + U_z_h) * 0.5f) * 0.5f + 0.5f;

	float W_x = W[i % hidden_size + hidden_size * 2 + hidden_size * 3 * (i / hidden_size)];
	float U_x = U[i % hidden_size + hidden_size * 2 + hidden_size * 3 * (i / hidden_size)];
	float h_bar = tanh(W_x + r * U_x);

	hidden[i] = z * hidden[i] + (1.f - z) * h_bar;
}


__global__ void relu(float *x, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size) return;

	if (x[i] < 0) x[i] = 0;
}


__global__ void initRandomState(curandState *state, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size) return;

	curand_init(i, 0, 0, &state[i]);
}


__global__ void addGumbel(float *x, curandState *state, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size) return;

	x[i] += -log(-log(curand_uniform(&state[i])));
}


__global__ void pairToKey(int *x, cub::KeyValuePair<int, float>* pair, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size) return;

	x[i] = pair[i].key;
}


void cudaErrorCheckUtil(cudaError_t error) {
	if (error != CUDA_SUCCESS)
	{
		throw sprintf(
			"[Error] %s (error code: %d) at %s line %d\n",
			cudaGetErrorString(error),
			error,
			__FILE__,
			__LINE__
		);
	}
}


const char *cublasGetErrorString(cublasStatus_t error)
{
	switch (error)
	{
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}


void cublasErrorCheckUtil(cublasStatus_t error) {
	if (error != CUBLAS_STATUS_SUCCESS)
	{
		throw sprintf(
			"[Error] %s (error code: %d) at %s line %d\n",
			cublasGetErrorString(error),
			error,
			__FILE__,
			__LINE__
		);
	}
}


void cudnnErrorCheckUtil(cudnnStatus_t error) {
	if (error != CUBLAS_STATUS_SUCCESS)
	{
		throw sprintf(
			"[Error] %s (error code: %d) at %s line %d\n",
			cudnnGetErrorString(error),
			error,
			__FILE__,
			__LINE__
		);
	}
}


void inference(
	int batch_size,
	int length,
	int local_size,
	int hidden_size,
	int embedding_size,
	int linear_hidden_size,
	int output_size,
	int* h_output,
	int* h_x,
	float* h_l_array,
	float* h_hidden,
	float* h_x_embedder_W,
	float* h_gru_xw,
	float* h_gru_xb,
	float* h_gru_hw,
	float* h_gru_hb,
	float* h_O1_W,
	float* h_O1_b,
	float* h_O2_W,
	float* h_O2_b
)
{
	// initialize
	std::cout << "initialize" << std::endl;
	auto x = ndarray<int>(batch_size, h_x);
	auto l_array = ndarray<float>(length, batch_size, local_size, h_l_array);
	auto hidden = ndarray<float>(batch_size, hidden_size, h_hidden);

	auto x_embedder_W = ndarray<float>(output_size, embedding_size, h_x_embedder_W);
	auto gru_xw = ndarray<float>(embedding_size + local_size, hidden_size * 3, h_gru_xw);
	auto gru_xb = ndarray<float>(hidden_size * 3, h_gru_xb);
	auto gru_hw = ndarray<float>(hidden_size, hidden_size * 3, h_gru_hw);
	auto gru_hb = ndarray<float>(hidden_size * 3, h_gru_hb);
	auto O1_W = ndarray<float>(hidden_size, linear_hidden_size, h_O1_W);
	auto O1_b = ndarray<float>(linear_hidden_size, h_O1_b);
	auto O2_W = ndarray<float>(linear_hidden_size, output_size, h_O2_W);
	auto O2_b = ndarray<float>(output_size, h_O2_b);

	auto gru_xb_b = ndarray<float>(batch_size, gru_xb.shape1);
	auto gru_hb_b = ndarray<float>(batch_size, gru_hb.shape1);
	auto O1_b_b = ndarray<float>(batch_size, O1_b.shape1);
	auto O2_b_b = ndarray<float>(batch_size, O2_b.shape1);

	auto xl = ndarray<float>(batch_size, embedding_size + local_size);
	auto w_gru_x = ndarray<float>(batch_size, hidden_size * 3);
	auto w_gru_h = ndarray<float>(batch_size, hidden_size * 3);
	auto w_out_x1 = ndarray<float>(batch_size, linear_hidden_size);
	auto w_out_x2 = ndarray<float>(batch_size, output_size);
	auto w_sampled = ndarray<cub::KeyValuePair<int, float>>(batch_size);

	auto gumbel_random_state = ndarray<curandState>(batch_size, output_size);

	int* h_pinned_output;
	cudaErrorCheckUtil(cudaHostAlloc(&h_pinned_output, length  * batch_size * sizeof(int), cudaHostAllocDefault));

	// create context
	std::cout << "create context" << std::endl;

	cudaStream_t stream;
	cudaErrorCheckUtil(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	cudaStream_t outputCopyStream;
	cudaErrorCheckUtil(cudaStreamCreateWithFlags(&outputCopyStream, cudaStreamNonBlocking));

	cudaEvent_t canToKey;
	cudaErrorCheckUtil(cudaEventCreateWithFlags(&canToKey, cudaEventDisableTiming));
	cudaErrorCheckUtil(cudaEventRecord(canToKey));

	cudaEvent_t canOutputCopy;
	cudaErrorCheckUtil(cudaEventCreateWithFlags(&canOutputCopy, cudaEventDisableTiming));

	cublasHandle_t cublasHandle;
	cublasErrorCheckUtil(cublasCreate(&cublasHandle));
	cublasErrorCheckUtil(cublasSetStream(cublasHandle, stream));

	for (int i = 0; i < batch_size; i++) {
		// broadcast
		cudaErrorCheckUtil(cudaMemcpyAsync(&gru_xb_b.device[i * gru_xb_b.shape2], gru_xb.device, gru_xb_b.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		cudaErrorCheckUtil(cudaMemcpyAsync(&gru_hb_b.device[i * gru_hb_b.shape2], gru_hb.device, gru_hb_b.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		cudaErrorCheckUtil(cudaMemcpyAsync(&O1_b_b.device[i * O1_b_b.shape2], O1_b.device, O1_b_b.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		cudaErrorCheckUtil(cudaMemcpyAsync(&O2_b_b.device[i * O2_b_b.shape2], O2_b.device, O2_b_b.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	}

	cudnnHandle_t cudnnHandle;
	cudnnErrorCheckUtil(cudnnCreate(&cudnnHandle));
	cudnnErrorCheckUtil(cudnnSetStream(cudnnHandle, stream));

	cudnnTensorDescriptor_t softmaxDesc;
	cudnnErrorCheckUtil(cudnnCreateTensorDescriptor(&softmaxDesc));
	cudnnErrorCheckUtil(cudnnSetTensor4dDescriptor(
		softmaxDesc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		w_out_x2.shape1,
		w_out_x2.shape2,
		1,
		1
	));
	initRandomState KERNEL_ARGS4(dim3(512), dim3(gumbel_random_state.size() / 512 + 1), 0, stream) (
		gumbel_random_state.device,  // curandState *state,
		gumbel_random_state.size()  // int size
		);

	int* h_argmax_offset = (int*)malloc((w_out_x2.shape1 + 1) * sizeof(int));
	for (int i = 0; i < w_out_x2.shape1 + 1; i++) {
		h_argmax_offset[i] = i * w_out_x2.shape2;
	}
	auto argmax_offset = ndarray<int>(w_out_x2.shape1 + 1, h_argmax_offset);

	size_t argmax_storage_bytes = 0;
	cudaErrorCheckUtil(cub::DeviceSegmentedReduce::ArgMax(
		NULL,  // void *d_temp_storage
		argmax_storage_bytes,  // size_t &temp_storage_bytes
		w_out_x2.device,  // InputIteratorT d_in
		w_sampled.device,  // OutputIteratorT d_out
		w_out_x2.shape1,  // int num_segments
		argmax_offset.device,  // OffsetIteratorT d_begin_offsets
		argmax_offset.device + 1,  // OffsetIteratorT d_end_offsets
		0,  // cudaStream_t stream
		true  // bool debug_synchronous
	));
	auto argmax_storage = ndarray<char>((int)argmax_storage_bytes);

	std::chrono::system_clock::time_point start, end;
	start = std::chrono::system_clock::now();
	for (int i_local = 0; i_local < length; i_local++) {
		// concat
		concat KERNEL_ARGS4(dim3(512), dim3(xl.size() / 512 + 1), 0, stream) (
			xl.device, // float* xl,
			x.device, // int* x,
			&l_array.device[i_local * (l_array.shape2 * l_array.shape3)], // float* l,
			x_embedder_W.device, // float* x_embedder_W,
			batch_size, // int batch_size,
			local_size, // int local_size,
			embedding_size // int embedding_size
			);

		// gru_x = prev_xl.dot(gru_xw) + gru_xb
		cudaErrorCheckUtil(cudaMemcpyAsync(w_gru_x.device, gru_xb_b.device, w_gru_x.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		float gemmAlpha = 1, gemmBeta = 1;
		cublasErrorCheckUtil(cublasSgemm(
			cublasHandle, // cublasHandle_t handle,
			CUBLAS_OP_N, // cublasOperation_t transa,
			CUBLAS_OP_N, // cublasOperation_t transb,
			gru_xw.shape2, // int m,
			xl.shape1, // int n,
			gru_xw.shape1, // int k,
			&gemmAlpha, // const float *alpha, /* host or device pointer */
			gru_xw.device, // const float *A,
			gru_xw.shape2, // int lda,
			xl.device, // const float *B,
			xl.shape2, // int ldb,
			&gemmBeta, // const float *beta, /* host or device pointer */
			w_gru_x.device, // float *C,
			w_gru_x.shape2 // int ldc
		));

		// gru_h = hidden.dot(gru_hw) + gru_hb
		cudaErrorCheckUtil(cudaMemcpyAsync(w_gru_h.device, gru_hb_b.device, w_gru_h.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		cublasErrorCheckUtil(cublasSgemm(
			cublasHandle, // cublasHandle_t handle,
			CUBLAS_OP_N, // cublasOperation_t transa,
			CUBLAS_OP_N, // cublasOperation_t transb,
			gru_hw.shape2, // int m,
			hidden.shape1, // int n,
			gru_hw.shape1, // int k,
			&gemmAlpha, // const float *alpha, /* host or device pointer */
			gru_hw.device, // const float *A,
			gru_hw.shape2, // int lda,
			hidden.device, // const float *B,
			hidden.shape2, // int ldb,
			&gemmBeta, // const float *beta, /* host or device pointer */
			w_gru_h.device, // float *C,
			w_gru_h.shape2 // int ldc
		));

		// gruElementWise
		gruElementWise KERNEL_ARGS4(dim3(512), dim3(hidden.size() / 512 + 1), 0, stream) (
			hidden.device,  // float* hidden
			w_gru_x.device,  // float* W
			w_gru_h.device,  // float* U
			batch_size,  // int batch_size
			hidden_size  // int hidden_size
			);

		// out_x = hidden.dot(O1_W) + O1_b
		cudaErrorCheckUtil(cudaMemcpyAsync(w_out_x1.device, O1_b_b.device, w_out_x1.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		cublasErrorCheckUtil(cublasSgemm(
			cublasHandle, // cublasHandle_t handle,
			CUBLAS_OP_N, // cublasOperation_t transa,
			CUBLAS_OP_N, // cublasOperation_t transb,
			O1_W.shape2, // int m,
			hidden.shape1, // int n,
			O1_W.shape1, // int k,
			&gemmAlpha, // const float *alpha, /* host or device pointer */
			O1_W.device, // const float *A,
			O1_W.shape2, // int lda,
			hidden.device, // const float *B,
			hidden.shape2, // int ldb,
			&gemmBeta, // const float *beta, /* host or device pointer */
			w_out_x1.device, // float *C,
			w_out_x1.shape2 // int ldc
		));

		// relu
		relu KERNEL_ARGS4(dim3(512), dim3(w_out_x1.size() / 512 + 1), 0, stream) (
			w_out_x1.device,  // float* x
			w_out_x1.size()  // int size
			);

		// out_x = out_x.dot(O2_W) + O2_b
		cudaErrorCheckUtil(cudaMemcpyAsync(w_out_x2.device, O2_b_b.device, w_out_x2.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		cublasErrorCheckUtil(cublasSgemm(
			cublasHandle, // cublasHandle_t handle,
			CUBLAS_OP_N, // cublasOperation_t transa,
			CUBLAS_OP_N, // cublasOperation_t transb,
			O2_W.shape2, // int m,
			w_out_x1.shape1, // int n,
			O2_W.shape1, // int k,
			&gemmAlpha, // const float *alpha, /* host or device pointer */
			O2_W.device, // const float *A,
			O2_W.shape2, // int lda,
			w_out_x1.device, // const float *B,
			w_out_x1.shape2, // int ldb,
			&gemmBeta, // const float *beta, /* host or device pointer */
			w_out_x2.device, // float *C,
			w_out_x2.shape2 // int ldc
		));

		// softmax
		auto dist = w_out_x2;
		float softmaxAlpha = 1, softmaxBeta = 0;
		cudnnErrorCheckUtil(cudnnSoftmaxForward(
			cudnnHandle, // cudnnHandle_t
			CUDNN_SOFTMAX_LOG, // cudnnSoftmaxAlgorithm_t
			CUDNN_SOFTMAX_MODE_CHANNEL, // cudnnSoftmaxMode_t
			&softmaxAlpha, // const void
			softmaxDesc, // const cudnnTensorDescriptor_t
			dist.device, // const void
			&softmaxBeta, // const void
			softmaxDesc, // const cudnnTensorDescriptor_t
			dist.device // void
		));

		// sampling
		addGumbel KERNEL_ARGS4(dim3(512), dim3(dist.size() / 512 + 1), 0, stream) (
			dist.device,  // float *x
			gumbel_random_state.device,  // curandState *state
			dist.size()  // int size
			);

		cudaErrorCheckUtil(cub::DeviceSegmentedReduce::ArgMax(
			argmax_storage.device,  // void *d_temp_storage
			argmax_storage_bytes,  // size_t &temp_storage_bytes
			dist.device,  // InputIteratorT d_in
			w_sampled.device,  // OutputIteratorT d_out
			dist.shape1,  // int num_segments
			argmax_offset.device,  // OffsetIteratorT d_begin_offsets
			argmax_offset.device + 1,  // OffsetIteratorT d_end_offsets
			stream,  // cudaStream_t stream
			false  // bool debug_synchronous
		));

		cudaStreamWaitEvent(stream, canToKey, 0);
		pairToKey KERNEL_ARGS4(dim3(512), dim3(x.size() / 512 + 1), 0, stream) (
			x.device,  // int *x
			w_sampled.device,  // cub::KeyValuePair<int, float>* pair
			x.size()  // int size
			);
		cudaEventRecord(canOutputCopy, stream);

		cudaStreamWaitEvent(outputCopyStream, canOutputCopy, 0);
		cudaMemcpyAsync(&h_pinned_output[i_local * batch_size], x.device, x.size() * sizeof(int), cudaMemcpyDeviceToHost, outputCopyStream);
		cudaEventRecord(canToKey, outputCopyStream);
	}

	cudaStreamSynchronize(stream);
	cudaStreamSynchronize(outputCopyStream);
	end = std::chrono::system_clock::now();

	double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000 / 1000;
	printf("time %lf[s]\n", time);

	cudaMemcpy(h_output, h_pinned_output, length * batch_size * sizeof(int), cudaMemcpyHostToHost);
}

//void main() {
//	inference(NULL);
//}
