#pragma once

extern "C" __declspec(dllexport) void inference(
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
);