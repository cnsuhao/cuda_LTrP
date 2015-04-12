//main.cpp
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <stdio.h>
#include <time.h>
#include <string>

using namespace cv;
using namespace cv::cuda;

//void match(const GpuMat& imageL, const GpuMat& imageR, GpuMat &image, int step, int radius, Stream& stream = Stream::Null());

void ltrp(const GpuMat& imageL, const GpuMat& imageR, GpuMat &resultL, GpuMat &resultR, int choose, Stream& stream = Stream::Null());

#define  N  37

int main()
{
	String OUTPATH = ".//result//";
	String imname[N] = { "im", "lbp", "im_cslbp", "im_csltp_t1",
		"im_csltp_t5", "im_csltp_t10", "im_csltp_2t1", "im_csltp_4t1", "im_csltp_6t1",
		"Baby1", "Baby2", "Baby3", "Bowling1", "Cloth1", "Flowerpots", "Rocks1",

		"Baby1_cslbp", "Baby2_cslbp", "Baby3_cslbp", "Bowling1_cslbp", "Cloth1_cslbp", "Flowerpots_cslbp", "Rocks1_cslbp",

		"Baby1_lbp", "Baby2_lbp", "Baby3_lbp", "Bowling1_lbp", "Cloth1_lbp", "Flowerpots_lbp", "Rocks1_lbp",

		"Baby1_csltp", "Baby2_csltp", "Baby3_csltp", "Bowling1_csltp", "Cloth1_csltp", "Flowerpots_csltp", "Rocks1_csltp"
	};

	String impath[N][2] = { { "im2.ppm", "im6.ppm" }, { ".//lbp//lbpl.png", ".//lbp//lbpr.png" },
	{ ".//cslbp//iml_cslbp.png", ".//cslbp//imr_cslbp.png" }, { ".//csltp/iml_csltp_t1.png", ".//csltp/imr_csltp_t1.png" },
	{ ".//csltp/iml_csltp_t5.png", ".//csltp/imr_csltp_t5.png" }, { ".//csltp/iml_csltp_t10.png", ".//csltp/imr_csltp_t10.png" },
	{ ".//csltp/iml_csltp_2t1.png", ".//csltp/imr_csltp_2t1.png" }, { ".//csltp/iml_csltp_4t1.png", ".//csltp/imr_csltp_4t1.png" },
	{ ".//csltp/iml_csltp_6t5.png", ".//csltp/imr_csltp_6t5.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Baby1//view1.png", "C://Users//dream//Desktop//ALL-2views//Baby1//view5.png" },
	{ "C://Users//dream//Desktop//ALL-2views//Baby2//view1.png", "C://Users//dream//Desktop//ALL-2views//Baby2//view5.png" },
	{ "C://Users//dream//Desktop//ALL-2views//Baby3//view1.png", "C://Users//dream//Desktop//ALL-2views//Baby3//view5.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Bowling1//view1.png", "C://Users//dream//Desktop//ALL-2views//Bowling1//view5.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Cloth1//view1.png", "C://Users//dream//Desktop//ALL-2views//Cloth1//view5.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Flowerpots//view1.png", "C://Users//dream//Desktop//ALL-2views//Flowerpots//view5.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Rocks1//view1.png", "C://Users//dream//Desktop//ALL-2views//Rocks1//view5.png" },


	{ "C://Users//dream//Desktop//ALL-2views//Cslbp//Baby1_cslbp_L.png", "C://Users//dream//Desktop//ALL-2views//Cslbp//Baby1_cslbp_R.png" },
	{ "C://Users//dream//Desktop//ALL-2views//Cslbp//Baby2_cslbp_L.png", "C://Users//dream//Desktop//ALL-2views//Cslbp//Baby2_cslbp_R.png" },
	{ "C://Users//dream//Desktop//ALL-2views//Cslbp//Baby3_cslbp_L.png", "C://Users//dream//Desktop//ALL-2views//Cslbp//Baby3_cslbp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Cslbp//Bowling1_cslbp_L.png", "C://Users//dream//Desktop//ALL-2views//Cslbp//Bowling1_cslbp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Cslbp//Cloth1_cslbp_L.png", "C://Users//dream//Desktop//ALL-2views//Cslbp//Cloth1_cslbp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Cslbp//Flowerpots_cslbp_L.png", "C://Users//dream//Desktop//ALL-2views//Cslbp//Flowerpots_cslbp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Cslbp//Rocks1_cslbp_L.png", "C://Users//dream//Desktop//ALL-2views//Cslbp//Rocks1_cslbp_R.png" },


	{ "C://Users//dream//Desktop//ALL-2views//Lbp//Baby1_lbp_L.png", "C://Users//dream//Desktop//ALL-2views//Lbp//Baby1_lbp_R.png" },
	{ "C://Users//dream//Desktop//ALL-2views//Lbp//Baby2_lbp_L.png", "C://Users//dream//Desktop//ALL-2views//Lbp//Baby2_lbp_R.png" },
	{ "C://Users//dream//Desktop//ALL-2views//Lbp//Baby3_lbp_L.png", "C://Users//dream//Desktop//ALL-2views//Lbp//Baby3_lbp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Lbp//Bowling1_lbp_L.png", "C://Users//dream//Desktop//ALL-2views//Lbp//Bowling1_lbp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Lbp//Cloth1_lbp_L.png", "C://Users//dream//Desktop//ALL-2views//Lbp//Cloth1_lbp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Lbp//Flowerpots_lbp_L.png", "C://Users//dream//Desktop//ALL-2views//Lbp//Flowerpots_lbp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Lbp//Rocks1_lbp_L.png", "C://Users//dream//Desktop//ALL-2views//Lbp//Rocks1_lbp_R.png" },


	{ "C://Users//dream//Desktop//ALL-2views//Csltp//Baby1_csltp_L.png", "C://Users//dream//Desktop//ALL-2views//Csltp//Baby1_csltp_R.png" },
	{ "C://Users//dream//Desktop//ALL-2views//Csltp//Baby2_csltp_L.png", "C://Users//dream//Desktop//ALL-2views//Csltp//Baby2_csltp_R.png" },
	{ "C://Users//dream//Desktop//ALL-2views//Csltp//Baby3_csltp_L.png", "C://Users//dream//Desktop//ALL-2views//Csltp//Baby3_csltp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Csltp//Bowling1_csltp_L.png", "C://Users//dream//Desktop//ALL-2views//Csltp//Bowling1_csltp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Csltp//Cloth1_csltp_L.png", "C://Users//dream//Desktop//ALL-2views//Csltp//Cloth1_csltp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Csltp//Flowerpots_csltp_L.png", "C://Users//dream//Desktop//ALL-2views//Csltp//Flowerpots_csltp_R.png" },

	{ "C://Users//dream//Desktop//ALL-2views//Csltp//Rocks1_csltp_L.png", "C://Users//dream//Desktop//ALL-2views//Csltp//Rocks1_csltp_R.png" },

	};



	for (int i = 0; i < N; i++){
		int at = 0;
		printf("N : %d imput at:", N);
		scanf("%d", &at);
		printf("\n");
		impath[at][0];

		Mat imageL = cv::imread(impath[at][0].c_str(), 0);
		Mat imageR = cv::imread(impath[at][1].c_str(), 0);

		//Mat imageL = cv::imread("im2.ppm", 0);
		//Mat imageR = cv::imread("im6.ppm", 0);
		//Mat imageL = cv::imread(".//lbp//lbpl.png", 0);
		//Mat imageR = cv::imread(".//lbp//lbpr.png", 0);
		//Mat imageL = cv::imread(".//cslbp//iml_cslbp.png", 0);
		//Mat imageR = cv::imread(".//cslbp//imr_cslbp.png", 0);
		//Mat imageL = cv::imread(".//csltp/iml_csltp_t1.png", 0);
		//Mat imageR = cv::imread(".//csltp/imr_csltp_t1.png", 0);
		//Mat imageL = cv::imread(".//csltp/iml_csltp_t5.png", 0);
		//Mat imageR = cv::imread(".//csltp/imr_csltp_t5.png", 0);
		//Mat imageL = cv::imread(".//csltp/iml_csltp_t10.png", 0);
		//Mat imageR = cv::imread(".//csltp/imr_csltp_t10.png", 0);

		//  	Mat imageL = cv::imread("iml.png", 0);
		//  	Mat imageR = cv::imread("imr.png", 0);
		Mat h_resultL, h_resultR;
		//	imshow("left", imageL);
		//	imshow("right", imageR);

		if (imageL.empty()) printf("read error\n");
		if (imageR.empty()) printf("read error\n");



		GpuMat gpuMatL, gpuMatR, d_resultL, d_resultR;
		int choose = 0;
		//int choose = 1;

		//int step, radius;
		//char strbuffer[256];
		String sL, sR;
		//printf("input step and radius\n");
		//while (scanf_s("%d %d", &step, &radius) && step != -1 || radius != -1){
		//sprintf_s(strbuffer, "_step%d_radius%d", step, radius);

		if (choose == 0){
			sL = "_csltrp_L";
			sR = "_csltrp_R";
		}
		else if (choose == 1){
			sL = "_ltrp_L";
			sR = "_ltrp_R";
		}
		clock_t startc, stopc;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		//gpuMatL.create(imageL.rows, imageL.cols, CV_8UC1);
		//gpuMatR.create(imageR.rows, imageR.cols, CV_8UC1);

		gpuMatL.upload(imageL);
		gpuMatR.upload(imageR);

		startc = clock();
		//cudaSetDevice(0);
		//match(gpuMatL, gpuMatR, output, step, radius);
		ltrp(gpuMatL, gpuMatR, d_resultL, d_resultR, choose);
		//lbp(gpuMatL, gpuMatR, d_resultL, d_resultR, radius);
		d_resultL.download(h_resultL);
		d_resultR.download(h_resultR);

		cudaThreadSynchronize();
		stopc = clock();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("%fs\n%fs\n", elapsedTime / 1000.0, (stopc - startc)*1.0 / CLOCKS_PER_SEC);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		sL = OUTPATH + imname[at] + sL + ".png";
		sR = OUTPATH + imname[at] + sR + ".png";
		//Mat result = image.clone();
		imwrite(sL.c_str(), h_resultL);
		imshow("result", h_resultL);

		imwrite(sR.c_str(), h_resultR);
		imshow("result", h_resultR);
		d_resultL.release();
		d_resultR.release();
		gpuMatL.release();
		gpuMatR.release();
		imageL.release();
		imageR.release();
	}
	//}
	waitKey(0);
	return 0;
}

extern "C" void testZNCC();

int main2(){
	testZNCC();
	return 0;
}