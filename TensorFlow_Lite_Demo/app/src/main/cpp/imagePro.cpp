//
// Created by panjq1 on 2017/10/22.
//
#include <string>
#include <android/log.h>
#include "opencv2/opencv.hpp"
#include "AndroidDebug.h"
#include "imagePro.h"
#include "com_panjq_opencv_alg_ImagePro.h"
#define LOG_TAG    "---JNILOG---" // 这个是自定义的LOG的标识
using namespace cv;
using namespace std;

//void printFloat(cv::Mat data, int maxRow);
void printFloat(cv::Mat data, int startRow ,int endRow,string name) ;
void splitMultiChannels(cv::Mat mat,cv::Mat &A,cv::Mat &B) ;
extern "C"
JNIEXPORT void JNICALL Java_com_panjq_opencv_alg_ImagePro_jniImagePro
        (JNIEnv *, jobject, jlong matAddrSrcImage, jlong matAddrDestImage, jlong transMatAddr){
    DEBUG__TIME0;
    Mat& srcImage  = *(Mat*)matAddrSrcImage;
    Mat& destImage = *(Mat*)matAddrDestImage;
    Mat& transMat = *(Mat*)transMatAddr;
    int width=srcImage.cols;
    int height=srcImage.rows;
    cv::cvtColor(srcImage,srcImage,CV_RGBA2RGB);

    cv::Mat A,B;
    splitMultiChannels(transMat, A, B);//分割为三通道
//    printFloat(A, 200,201,"A");
//    printFloat(B, 200,201,"B");

//    cv::cvtColor(A,A,CV_RGB2BGR);

    LOGE("A,shspe:[%d,%d,%d]",A.rows,A.cols,A.channels());
    LOGE("B,shspe:[%d,%d,%d]",B.rows,B.cols,B.channels());

    cv::resize(A,A,cv::Size(width,height));
    cv::resize(B,B,cv::Size(width,height));

    LOGE("A,resize shspe:[%d,%d,%d]",A.rows,A.cols,A.channels());
    LOGE("B,resize shspe:[%d,%d,%d]",B.rows,B.cols,B.channels());
    LOGE("srcImage,shspe:[%d,%d,%d]",srcImage.rows,srcImage.cols,srcImage.channels());
    //LOGE("A,data,shspe:[%d,%d,%d]",A.at(100,100)[0],A.at(100,100)[1],A.at(100,100)[2]);

    srcImage.convertTo(srcImage,CV_32FC3,1/255.0,0);
    // destImage=A.mul(srcImage);cv::add(destImage,B,destImage);
    destImage=A.mul(srcImage)+B;
    destImage.convertTo(destImage,CV_8UC3,255,0);

    A.release();
    B.release();
    //blur(srcImage,destImage,Size(20,20));
    LOGE("destImage,shspe:[%d,%d,%d]",destImage.rows,destImage.cols,destImage.channels());
    cv::cvtColor(destImage,destImage,CV_RGB2RGBA);
    DEBUG__TIME1;
    LOGE("Run time:jniImagePro3=%dms\n",(TIME1-TIME0)/1000);
   // LOGE("jniImagePro3: ouput image size=[%d,%d],channels=%d\n",destImage.rows,destImage.cols,destImage.channels());
}

/*
	将多个mat合并为多通道mat
*/
cv::Mat mergeMultiChannels(cv::Mat A,cv::Mat B) {
    cv::Mat AB;
    vector<cv::Mat> ABchannels;
    ABchannels.push_back(A);
    ABchannels.push_back(B);
    cv::merge(ABchannels, AB);
    return AB;
}

/*
	将6通道的mat分割成2个三通道的mat
*/
void splitMultiChannels(cv::Mat mat,cv::Mat &A,cv::Mat &B) {
    vector<cv::Mat> channels;
    cv::split(mat, channels);//分割image1的通
    vector<cv::Mat> Avec, Bvec;
    Avec.push_back(channels[0]);
    Avec.push_back(channels[1]);
    Avec.push_back(channels[2]);

    Bvec.push_back(channels[3]);
    Bvec.push_back(channels[4]);
    Bvec.push_back(channels[5]);

    cv::merge(Avec, A);
    cv::merge(Bvec, B);
}


void printFloat(cv::Mat data, int startRow ,int endRow,string name) {
    for (int i = startRow; i < endRow; i++)
    {
        float* data_preRow = data.ptr<float>(i);
        for (int j = 0; j < data.cols; j++)
        {
           // printf("RGB:[%f,%f,%f]", data_preRow[j], data_preRow[j + 1], data_preRow[j + 2]);
            LOGE("%s,RGB:[%f,%f,%f]", name.c_str(),data_preRow[j], data_preRow[j + 1], data_preRow[j + 2]);

        }
        LOGE("\n");
    }
}
