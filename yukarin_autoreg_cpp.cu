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

#include "yukarin_autoreg_cpp.h"


// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}
#endif

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error) {
	switch (error) {
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

	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";

	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
	}

	return "<unknown>";
}
#endif

#ifdef CUDNN_H_
// cuDNN API errors
static const char *_cudaGetErrorEnum(cudnnStatus_t error) {
	return cudnnGetErrorString(error);
}
#endif

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		exit(EXIT_FAILURE);
	}
}

#ifdef __DRIVER_TYPES_H__
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#endif


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
	checkCudaErrors(cudaMalloc(&x, size * sizeof(T)));
	if (h != NULL) {
		checkCudaErrors(cudaMemcpy(x, h, size * sizeof(T), cudaMemcpyHostToDevice));
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
	ndarray(T* h = NULL) : host(h) {
	}
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

__global__ void floatToDouble(float *src, double *dst, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	dst[i] = (double) src[i];
}


__global__ void addGumbel(double *x, curandState *state, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size) return;

	x[i] += -log(-log(curand_uniform(&state[i])));
}


__global__ void pairToKey(int *x, cub::KeyValuePair<int, double>* pair, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size) return;

	x[i] = pair[i].key;
}


auto g_x = ndarray<int>();
auto g_l_array = ndarray<float>();
auto g_hidden = ndarray<float>();

int* g_h_pinned_output;

cudaStream_t g_stream;

cudaGraphExec_t g_graphExec;

int g_graph_length;

void initialize(
	int graph_length,
	int max_batch_size,
	int local_size,
	int hidden_size,
	int embedding_size,
	int linear_hidden_size,
	int output_size,
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
	int* h_pinned_x;
	checkCudaErrors(cudaHostAlloc(&h_pinned_x, max_batch_size * sizeof(int), cudaHostAllocDefault));

	float* h_pinned_l_array;
	checkCudaErrors(cudaHostAlloc(&h_pinned_l_array, graph_length  * max_batch_size * local_size * sizeof(float), cudaHostAllocDefault));

	float* h_pinned_hidden;
	checkCudaErrors(cudaHostAlloc(&h_pinned_hidden, max_batch_size * hidden_size * sizeof(float), cudaHostAllocDefault));

	checkCudaErrors(cudaHostAlloc(&g_h_pinned_output, graph_length  * max_batch_size * sizeof(int), cudaHostAllocDefault));

	auto x = ndarray<int>(max_batch_size, h_pinned_x);
	auto l_array = ndarray<float>(graph_length, max_batch_size, local_size, h_pinned_l_array);
	auto hidden = ndarray<float>(max_batch_size, hidden_size, h_pinned_hidden);

	auto x_embedder_W = ndarray<float>(output_size, embedding_size, h_x_embedder_W);
	auto gru_xw = ndarray<float>(embedding_size + local_size, hidden_size * 3, h_gru_xw);
	auto gru_xb = ndarray<float>(hidden_size * 3, h_gru_xb);
	auto gru_hw = ndarray<float>(hidden_size, hidden_size * 3, h_gru_hw);
	auto gru_hb = ndarray<float>(hidden_size * 3, h_gru_hb);
	auto O1_W = ndarray<float>(hidden_size, linear_hidden_size, h_O1_W);
	auto O1_b = ndarray<float>(linear_hidden_size, h_O1_b);
	auto O2_W = ndarray<float>(linear_hidden_size, output_size, h_O2_W);
	auto O2_b = ndarray<float>(output_size, h_O2_b);

	auto gru_xb_b = ndarray<float>(max_batch_size, gru_xb.shape1);
	auto gru_hb_b = ndarray<float>(max_batch_size, gru_hb.shape1);
	auto O1_b_b = ndarray<float>(max_batch_size, O1_b.shape1);
	auto O2_b_b = ndarray<float>(max_batch_size, O2_b.shape1);

	auto xl = ndarray<float>(max_batch_size, embedding_size + local_size);
	auto w_gru_x = ndarray<float>(max_batch_size, hidden_size * 3);
	auto w_gru_h = ndarray<float>(max_batch_size, hidden_size * 3);
	auto w_out_x1 = ndarray<float>(max_batch_size, linear_hidden_size);
	auto w_out_x2 = ndarray<float>(max_batch_size, output_size);
	auto w_dist = ndarray<double>(max_batch_size, output_size);
	auto w_sampled = ndarray<cub::KeyValuePair<int, double>>(max_batch_size);

	auto gumbel_random_state = ndarray<curandState>(max_batch_size, output_size);

	// create context
	std::cout << "create context" << std::endl;

	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

	cudaStream_t biasCopyStream;
	checkCudaErrors(cudaStreamCreateWithFlags(&biasCopyStream, cudaStreamDefault));

	cudaStream_t hiddenStream;
	checkCudaErrors(cudaStreamCreateWithFlags(&hiddenStream, cudaStreamDefault));

	cudaStream_t outputCopyStream;
	checkCudaErrors(cudaStreamCreateWithFlags(&outputCopyStream, cudaStreamDefault));

	cublasHandle_t cublasHandle;
	checkCudaErrors(cublasCreate(&cublasHandle));
	checkCudaErrors(cublasSetStream(cublasHandle, stream));

	cublasHandle_t cublasHiddenHandle;
	checkCudaErrors(cublasCreate(&cublasHiddenHandle));
	checkCudaErrors(cublasSetStream(cublasHiddenHandle, hiddenStream));

	for (int i = 0; i < max_batch_size; i++) {
		// broadcast
		checkCudaErrors(cudaMemcpyAsync(&gru_xb_b.device[i * gru_xb_b.shape2], gru_xb.device, gru_xb_b.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		checkCudaErrors(cudaMemcpyAsync(&gru_hb_b.device[i * gru_hb_b.shape2], gru_hb.device, gru_hb_b.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		checkCudaErrors(cudaMemcpyAsync(&O1_b_b.device[i * O1_b_b.shape2], O1_b.device, O1_b_b.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		checkCudaErrors(cudaMemcpyAsync(&O2_b_b.device[i * O2_b_b.shape2], O2_b.device, O2_b_b.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	}

	cudnnHandle_t cudnnHandle;
	checkCudaErrors(cudnnCreate(&cudnnHandle));
	checkCudaErrors(cudnnSetStream(cudnnHandle, stream));

	initRandomState KERNEL_ARGS4(dim3(gumbel_random_state.size() / 512 + 1), dim3(512), 0, stream) (
		gumbel_random_state.device,  // curandState *state,
		gumbel_random_state.size()  // int size
		);

	int* h_argmax_offset = (int*)malloc((w_out_x2.shape1 + 1) * sizeof(int));
	for (int i = 0; i < w_out_x2.shape1 + 1; i++) {
		h_argmax_offset[i] = i * w_out_x2.shape2;
	}
	auto argmax_offset = ndarray<int>(w_out_x2.shape1 + 1, h_argmax_offset);

	size_t argmax_storage_bytes = 0;
	checkCudaErrors(cub::DeviceSegmentedReduce::ArgMax(
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

	// graph
	cudaEvent_t copyHiddenDone;
	checkCudaErrors(cudaEventCreateWithFlags(&copyHiddenDone, cudaEventDisableTiming));

	cudaEvent_t elementWiseDone, gemmO2Done, argmaxDone;
	checkCudaErrors(cudaEventCreateWithFlags(&elementWiseDone, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&gemmO2Done, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&argmaxDone, cudaEventDisableTiming));

	cudaEvent_t copyGruXbDone, copyGruHbDone, copyO1bDone, copyO2bDone;
	checkCudaErrors(cudaEventCreateWithFlags(&copyGruXbDone, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&copyGruHbDone, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&copyO1bDone, cudaEventDisableTiming));
	checkCudaErrors(cudaEventCreateWithFlags(&copyO2bDone, cudaEventDisableTiming));

	cudaEvent_t gemmO1Done;
	checkCudaErrors(cudaEventCreateWithFlags(&gemmO1Done, cudaEventDisableTiming));

	cudaEvent_t gemmGruHDone;
	checkCudaErrors(cudaEventCreateWithFlags(&gemmGruHDone, cudaEventDisableTiming));

	cudaEvent_t outputCopyDone;
	checkCudaErrors(cudaEventCreateWithFlags(&outputCopyDone, cudaEventDisableTiming));

	cudaEvent_t toKeyDone;
	checkCudaErrors(cudaEventCreateWithFlags(&toKeyDone, cudaEventDisableTiming));

	cudnnTensorDescriptor_t softmaxDesc;
	checkCudaErrors(cudnnCreateTensorDescriptor(&softmaxDesc));
	checkCudaErrors(cudnnSetTensor4dDescriptor(
		softmaxDesc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_DOUBLE,
		max_batch_size,
		w_out_x2.shape2,
		1,
		1
	));

	std::cout << "graph start" << std::endl;

	checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
	cudaEventRecord(elementWiseDone, stream);  // for joining

	checkCudaErrors(cudaMemcpyAsync(x.device, x.host, max_batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(l_array.device, l_array.host, l_array.shape1 * max_batch_size * l_array.shape3 * sizeof(float), cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(hidden.device, hidden.host, max_batch_size * hidden.shape2 * sizeof(float), cudaMemcpyHostToDevice, stream));
	cudaEventRecord(copyHiddenDone, stream);
	cudaStreamWaitEvent(hiddenStream, copyHiddenDone, 0);

	for (int i_local = 0; i_local < graph_length; i_local++) {
		// concat
		concat KERNEL_ARGS4(dim3(512), dim3(max_batch_size * xl.shape2 / 512 + 1), 0, stream) (
			xl.device, // float* xl,
			x.device, // int* x,
			&l_array.device[i_local * (max_batch_size * l_array.shape3)], // float* l,
			x_embedder_W.device, // float* x_embedder_W,
			max_batch_size, // int batch_size,
			local_size, // int local_size,
			embedding_size // int embedding_size
			);

		// gru_x = prev_xl.dot(gru_xw) + gru_xb
		cudaStreamWaitEvent(biasCopyStream, elementWiseDone, 0);
		checkCudaErrors(cudaMemcpyAsync(w_gru_x.device, gru_xb_b.device, max_batch_size * w_gru_x.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, biasCopyStream));
		cudaEventRecord(copyGruXbDone, biasCopyStream);

		float gemmAlpha = 1, gemmBeta = 1;
		cudaStreamWaitEvent(stream, copyGruXbDone, 0);
		checkCudaErrors(cublasSgemm(
			cublasHandle, // cublasHandle_t handle,
			CUBLAS_OP_N, // cublasOperation_t transa,
			CUBLAS_OP_N, // cublasOperation_t transb,
			gru_xw.shape2, // int m,
			max_batch_size, // int n,
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
		checkCudaErrors(cudaStreamWaitEvent(biasCopyStream, elementWiseDone, 0));
		checkCudaErrors(cudaMemcpyAsync(w_gru_h.device, gru_hb_b.device, max_batch_size * w_gru_h.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, biasCopyStream));
		checkCudaErrors(cudaEventRecord(copyGruHbDone, biasCopyStream));

		checkCudaErrors(cudaStreamWaitEvent(hiddenStream, copyGruHbDone, 0));
		checkCudaErrors(cudaStreamWaitEvent(hiddenStream, gemmO1Done, 0));
		checkCudaErrors(cublasSgemm(
			cublasHiddenHandle, // cublasHandle_t handle,
			CUBLAS_OP_N, // cublasOperation_t transa,
			CUBLAS_OP_N, // cublasOperation_t transb,
			gru_hw.shape2, // int m,
			max_batch_size, // int n,
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
		checkCudaErrors(cudaEventRecord(gemmGruHDone, hiddenStream));

		// gruElementWise
		cudaStreamWaitEvent(stream, gemmGruHDone, 0);
		gruElementWise KERNEL_ARGS4(dim3(max_batch_size * hidden.shape2 / 512 + 1), dim3(512), 0, stream) (
			hidden.device,  // float* hidden
			w_gru_x.device,  // float* W
			w_gru_h.device,  // float* U
			max_batch_size,  // int batch_size
			hidden_size  // int hidden_size
			);
		cudaEventRecord(elementWiseDone, stream);

		// out_x = hidden.dot(O1_W) + O1_b
		cudaStreamWaitEvent(biasCopyStream, gemmO2Done, 0);
		checkCudaErrors(cudaMemcpyAsync(w_out_x1.device, O1_b_b.device, max_batch_size * w_out_x1.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, biasCopyStream));
		cudaEventRecord(copyO1bDone, biasCopyStream);

		cudaStreamWaitEvent(stream, copyO1bDone, 0);
		checkCudaErrors(cublasSgemm(
			cublasHandle, // cublasHandle_t handle,
			CUBLAS_OP_N, // cublasOperation_t transa,
			CUBLAS_OP_N, // cublasOperation_t transb,
			O1_W.shape2, // int m,
			max_batch_size, // int n,
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
		cudaEventRecord(gemmO1Done, stream);

		// relu
		relu KERNEL_ARGS4(dim3(max_batch_size * w_out_x1.shape2 / 512 + 1), dim3(512), 0, stream) (
			w_out_x1.device,  // float* x
			max_batch_size * w_out_x1.shape2  // int size
			);

		// out_x = out_x.dot(O2_W) + O2_b
		cudaStreamWaitEvent(biasCopyStream, argmaxDone, 0);
		checkCudaErrors(cudaMemcpyAsync(w_out_x2.device, O2_b_b.device, max_batch_size * w_out_x2.shape2 * sizeof(float), cudaMemcpyDeviceToDevice, biasCopyStream));
		cudaEventRecord(copyO2bDone, biasCopyStream);

		cudaStreamWaitEvent(stream, copyO2bDone, 0);
		checkCudaErrors(cublasSgemm(
			cublasHandle, // cublasHandle_t handle,
			CUBLAS_OP_N, // cublasOperation_t transa,
			CUBLAS_OP_N, // cublasOperation_t transb,
			O2_W.shape2, // int m,
			max_batch_size, // int n,
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
		cudaEventRecord(gemmO2Done, stream);

		floatToDouble KERNEL_ARGS4(dim3(max_batch_size * w_out_x2.shape2 / 512 + 1), dim3(512), 0, stream) (
			w_out_x2.device, // float *src
			w_dist.device, // double *dst
			max_batch_size * w_out_x2.shape2 // int size
		);

		// softmax
		double softmaxAlpha = 1, softmaxBeta = 0;
		checkCudaErrors(cudnnSoftmaxForward(
			cudnnHandle, // cudnnHandle_t
			CUDNN_SOFTMAX_LOG, // cudnnSoftmaxAlgorithm_t
			CUDNN_SOFTMAX_MODE_CHANNEL, // cudnnSoftmaxMode_t
			&softmaxAlpha, // const void
			softmaxDesc, // const cudnnTensorDescriptor_t
			w_dist.device, // const void
			&softmaxBeta, // const void
			softmaxDesc, // const cudnnTensorDescriptor_t
			w_dist.device // void
		));

		// sampling
		addGumbel KERNEL_ARGS4(dim3(max_batch_size * w_dist.shape2 / 512 + 1), dim3(512), 0, stream) (
			w_dist.device,  // double *x
			gumbel_random_state.device,  // curandState *state
			max_batch_size * w_dist.shape2  // int size
			);

		checkCudaErrors(cub::DeviceSegmentedReduce::ArgMax(
			argmax_storage.device,  // void *d_temp_storage
			argmax_storage_bytes,  // size_t &temp_storage_bytes
			w_dist.device,  // InputIteratorT d_in
			w_sampled.device,  // OutputIteratorT d_out
			max_batch_size,  // int num_segments
			argmax_offset.device,  // OffsetIteratorT d_begin_offsets
			argmax_offset.device + 1,  // OffsetIteratorT d_end_offsets
			stream,  // cudaStream_t stream
			false  // bool debug_synchronous
		));
		cudaEventRecord(argmaxDone, stream);

		cudaStreamWaitEvent(stream, outputCopyDone, 0);
		pairToKey KERNEL_ARGS4(dim3(max_batch_size / 512 + 1), dim3(512), 0, stream) (
			x.device,  // int *x
			w_sampled.device,  // cub::KeyValuePair<int, double>* pair
			max_batch_size  // int size
			);
		cudaEventRecord(toKeyDone, stream);

		cudaStreamWaitEvent(outputCopyStream, toKeyDone, 0);
		cudaMemcpyAsync(&g_h_pinned_output[i_local * max_batch_size], x.device, max_batch_size * sizeof(int), cudaMemcpyDeviceToHost, outputCopyStream);
		cudaEventRecord(outputCopyDone, outputCopyStream);
	}

	cudaMemcpyAsync(x.host, x.device, max_batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(hidden.host, hidden.device, max_batch_size * hidden.shape2 * sizeof(float), cudaMemcpyDeviceToHost, stream);

	cudaStreamWaitEvent(stream, outputCopyDone, 0);

	cudaGraph_t graph;
	checkCudaErrors(cudaStreamEndCapture(stream, &graph));

	checkCudaErrors(cudaGraphInstantiate(&g_graphExec, graph, NULL, NULL, 0));

	checkCudaErrors(cudaGraphDestroy(graph));
	std::cout << "graph done" << std::endl;

	// destroy
	checkCudaErrors(cudaEventDestroy(elementWiseDone));
	checkCudaErrors(cudaEventDestroy(gemmO2Done));
	checkCudaErrors(cudaEventDestroy(argmaxDone));
	checkCudaErrors(cudaEventDestroy(copyGruXbDone));
	checkCudaErrors(cudaEventDestroy(copyGruHbDone));
	checkCudaErrors(cudaEventDestroy(copyO1bDone));
	checkCudaErrors(cudaEventDestroy(copyO2bDone));
	checkCudaErrors(cudaEventDestroy(gemmO1Done));
	checkCudaErrors(cudaEventDestroy(gemmGruHDone));
	checkCudaErrors(cudaEventDestroy(outputCopyDone));
	checkCudaErrors(cudaEventDestroy(toKeyDone));

	// destroy
	checkCudaErrors(cudaStreamDestroy(biasCopyStream));
	checkCudaErrors(cudaStreamDestroy(hiddenStream));
	checkCudaErrors(cudaStreamDestroy(outputCopyStream));

	// global parameters
	g_x = x;
	g_l_array = l_array;
	g_hidden = hidden;

	g_stream = stream;

	g_graph_length = graph_length;
}


void inference(
	int batch_size,
	int length,
	int* h_output,
	int* h_x,
	float* h_l_array,
	float* h_hidden
)
{
	std::chrono::system_clock::time_point start, end;
	start = std::chrono::system_clock::now();

	checkCudaErrors(cudaMemcpyAsync(g_x.host, h_x, batch_size * sizeof(int), cudaMemcpyHostToHost, g_stream));
	checkCudaErrors(cudaMemcpyAsync(g_hidden.host, h_hidden, batch_size * g_hidden.shape2 * sizeof(float), cudaMemcpyHostToHost, g_stream));

	int l_size = batch_size * g_l_array.shape3;

	int max_batch_size = g_l_array.shape2;
	int g_l_size = g_l_array.shape2 * g_l_array.shape3;

	int now_length = 0;
	while (now_length < length) {
		// re-zero
		checkCudaErrors(cudaMemsetAsync(g_l_array.host, 0, g_l_array.size(), g_stream));

		// choice graph length
		int next_length;
		if (length - now_length >= g_graph_length) {
			next_length = g_graph_length;
		}
		else {
			next_length = length - now_length;
		}

		// forward
		for (int i = 0; i < next_length; i++) {
			checkCudaErrors(cudaMemcpyAsync(&g_l_array.host[i * g_l_size], &h_l_array[(now_length + i) * l_size], l_size * sizeof(float), cudaMemcpyHostToHost, g_stream));
		}

		checkCudaErrors(cudaGraphLaunch(g_graphExec, g_stream));

		for (int i = 0; i < next_length; i++) {
			checkCudaErrors(cudaMemcpyAsync(&h_output[(now_length + i) * batch_size], &g_h_pinned_output[i * max_batch_size], batch_size * sizeof(int), cudaMemcpyHostToHost, g_stream));
		}

		// next loop
		now_length += next_length;
	}

	checkCudaErrors(cudaMemcpyAsync(h_x, g_x.host, batch_size * sizeof(int), cudaMemcpyHostToHost, g_stream));
	checkCudaErrors(cudaMemcpyAsync(h_hidden, g_hidden.host, batch_size * g_hidden.shape2 * sizeof(float), cudaMemcpyHostToHost, g_stream));

	end = std::chrono::system_clock::now();

	double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000 / 1000;
	printf("time %lf[s]\n", time);
}
