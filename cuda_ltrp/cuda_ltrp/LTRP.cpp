//SSD.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

using namespace cv;
using namespace cv::cuda;

//void match_caller(const PtrStepSz<uchar> imageL, const PtrStepSz<uchar> imageR, PtrStepSz<uchar> image, int step, int radius, cudaStream_t stream);

//extern "C" void match_caller(const PtrStepSz<uchar> imageL, const PtrStepSz<uchar> imageR, PtrStepSz<uchar> d_image, int step, int radius, cudaStream_t stream);

extern "C" void ltrp_caller(const PtrStepSz<uchar> imageL, const PtrStepSz<uchar> imageR, PtrStepSz<uchar> resultL, PtrStepSz<uchar> resultR, int choose, cudaStream_t stream);


void ltrp(const GpuMat& imageL, const GpuMat& imageR, GpuMat &resultL, GpuMat &resultR, int choose, Stream& stream = Stream::Null())
{
	CV_Assert(imageL.type() == CV_8UC1);
	CV_Assert(imageR.type() == CV_8UC1);
	resultL.create(imageL.size(), CV_8UC1);
	resultR.create(imageR.size(), CV_8UC1);
	cudaStream_t s = StreamAccessor::getStream(stream);
	ltrp_caller(imageL, imageR, resultL, resultR, choose, s);
}


