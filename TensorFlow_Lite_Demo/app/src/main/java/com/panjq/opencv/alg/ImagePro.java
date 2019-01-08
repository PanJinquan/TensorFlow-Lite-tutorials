package com.panjq.opencv.alg;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.media.ThumbnailUtils;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.Utils;

/**
 * Created by panjq1 on 2017/10/22.
 */

public class ImagePro {

    //定义tflite输入输出数组维度
    private static final int BATCH_SIZE = 1;
    private static final int IMAGE_HEIGHT = 128;
    private static final int IMAGE_WIDTH = 128;
    private static final int INPUTS_CHANNEL = 3;
    private static final int OUTPUT_CHANNEL = 6;
    private  float[][][][] tf_input = new float[BATCH_SIZE][IMAGE_HEIGHT][IMAGE_WIDTH] [INPUTS_CHANNEL];//tflite输入维度
    private  float[][][][] tf_output = new float[BATCH_SIZE][IMAGE_HEIGHT][IMAGE_WIDTH] [OUTPUT_CHANNEL];//tflite输出维度

    private static final String    TAG = "ImagePro:";
    static {
        System.loadLibrary("imagePro-lib");
        //System.loadLibrary("tensorflowlite_jni");
    }

    /**
     * 调用Tensorflow lite实现图像处理
     * @param origImage 原始Bitmap图像
     * @param context
     * @return 返回处理后的Bitmap图像
     */
    public Bitmap ImageProcessing(Bitmap origImage, Context context) {
        Log.i(TAG, "origImage width,height "+origImage.getWidth()+","+origImage.getHeight());
       // jniImagePro3(origMat.getNativeObjAddr(), destMat.getNativeObjAddr());
        Bitmap destBitmap=EnhancerImage(origImage,context);
        Log.i(TAG, "destBitmap width,height "+destBitmap.getWidth()+","+destBitmap.getHeight());
        return destBitmap;
    }


    /**
     * 调用Tensorflow lite实现图像增强
     * @param origImage 原始Bitmap图像
     * @param context
     * @return 返回处理后的Bitmap图像
     */
    public Bitmap EnhancerImage(Bitmap origImage,Context context){
        //对输入的图片进行缩放
        Bitmap resize_bitmat=scaleBitmap(origImage, IMAGE_WIDTH, IMAGE_HEIGHT);
        tf_input = get4Pixels(resize_bitmat);
        //tflite模型放在assets中
//        String model_path= "y_net_float_256.tflite";
        String model_path= "optimize_graph_float_128.tflite";
        //初始化TensorflowLite
        TensorflowLite mTFlite =new TensorflowLite(context,model_path);
        //mTFlite.setNumThreads(4);
        mTFlite.setUseNNAPI(true);

        //运行runTFlite
        float[] tranMat=null;
        try {
            long startTime = System.currentTimeMillis(); //起始时间
            mTFlite.runTFlite(tf_input,tf_output);
            long endTime = System.currentTimeMillis(); //结束时间
            Log.i(TAG, String.format("CNN Run time: %d ms", (endTime - startTime)));
//            tranMat= convertArr3ChannelTO1D(tf_output);
            tranMat= convertArrMultiChannelTO1D(tf_output);


        } catch (IOException e) {
            e.printStackTrace();
        }
        //调用Opencv进行后续处理
        int w=origImage.getWidth();
        int h=origImage.getHeight();
        Mat origMat = new Mat();
        Mat destMat = new Mat();
        Mat cv_tranMat = new Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CvType.CV_32FC(6));
//        Mat cv_tranMat = new Mat(256, 256, CvType.CV_32FC3);
        cv_tranMat.put(0,0,tranMat);
        Utils.bitmapToMat(origImage, origMat);//Bitmap转OpenCV的Mat
        jniImagePro(origMat.getNativeObjAddr(), destMat.getNativeObjAddr(),cv_tranMat.getNativeObjAddr());
        Bitmap bitImage = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(destMat, bitImage);//OpenCV的Mat转Bitmap显示
        return bitImage;
    }


    /**
     * 保存图片
     * @param bmp
     * @param name
     */

    public static void saveImage(Bitmap bmp,String name) {
        File appDir = new File(Environment.getExternalStorageDirectory(), "OpencvDemo");
        if (!appDir.exists()) {
            appDir.mkdir();
        }
        String fileName = name;
        File file = new File(appDir, fileName);
//        if (file.exists()) {
//            file.delete();
//        }
        try {
            FileOutputStream fos = new FileOutputStream(file);
            bmp.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();
            Log.e(TAG, "图片保存成功...");
        } catch (FileNotFoundException e) {
            Log.e(TAG, "图片保存失败...");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 根据给定的宽和高进行拉伸
     *
     * @param origin    原图
     * @param newWidth  新图的宽
     * @param newHeight 新图的高
     * @return new Bitmap
     */
    private Bitmap scaleBitmap(Bitmap origin, int newWidth, int newHeight) {
        if (origin == null) {
            return null;
        }
        int height = origin.getHeight();
        int width = origin.getWidth();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();

        matrix.postScale(scaleWidth, scaleHeight);// 使用后乘
        Bitmap newBM = Bitmap.createBitmap(origin, 0, 0, width, height, matrix, false);
//        if (!origin.isRecycled()) {
//            origin.recycle();
//        }
        return newBM;
    }



    /**
     *  获得Bitmap像素，用一维数组保存
     * @param bitmap
     * @return
     */
    private float[] getPixels(Bitmap bitmap) {
        int[] intValues = new int[IMAGE_HEIGHT * IMAGE_WIDTH];
        float[] floatValues = new float[IMAGE_HEIGHT * IMAGE_WIDTH * INPUTS_CHANNEL];
        if (bitmap.getWidth() != IMAGE_WIDTH || bitmap.getHeight() != IMAGE_HEIGHT) {
            // rescale the bitmap if needed
            bitmap = ThumbnailUtils.extractThumbnail(bitmap, IMAGE_HEIGHT, IMAGE_WIDTH);
        }
        bitmap.getPixels(intValues,0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = Color.red(val) / 255.0f;
            floatValues[i * 3 + 1] = Color.green(val) / 255.0f;
            floatValues[i * 3 + 2] = Color.blue(val) / 255.0f;
        }
        return floatValues;
    }

    /**
     * 获得bitmap像素，用4维数组保存
     * @param bitmap
     * @return
     */
    private float[][][][] get4Pixels(Bitmap bitmap) {
        int[] intValues = new int[IMAGE_HEIGHT * IMAGE_WIDTH];
        if (bitmap.getWidth() != IMAGE_WIDTH || bitmap.getHeight() != IMAGE_HEIGHT) {
            // rescale the bitmap if needed
            bitmap = ThumbnailUtils.extractThumbnail(bitmap, IMAGE_HEIGHT, IMAGE_WIDTH);
        }
        bitmap.getPixels(intValues,0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < IMAGE_WIDTH; ++i) {
            for (int j = 0; j < IMAGE_HEIGHT; ++j) {
                final int val = intValues[i* IMAGE_WIDTH +j];
                tf_input[0][i][j][0] = Color.red(val) / 255.0f;
                tf_input[0][i][j][1] = Color.green(val) / 255.0f;
                tf_input[0][i][j][2] = Color.blue(val) / 255.0f;
            }
        }
        return tf_input;
    }

    /**
     * 将3通道的4维数组转为单通道的1维数组
     * @param inputdata
     * @return
     */
    public float[] convertArr3ChannelTO1D(float[][][][] inputdata){
        int index=0;
        float[] dest = new float[IMAGE_HEIGHT * IMAGE_WIDTH * 3];
        for (int i = 0; i < IMAGE_WIDTH; ++i) {
            for (int j = 0; j < IMAGE_HEIGHT; ++j) {
                index=(i* IMAGE_WIDTH +j);
                dest[index* 3]=inputdata[0][i][j][0];
                dest[index * 3+1]=inputdata[0][i][j][1];
                dest[index * 3+2]=inputdata[0][i][j][2];
            }
        }
        return dest;
    }

    /**
     * 将多通道的4维数组转为单通道的1维数组
     * @param inputdata
     * @return
     */
    public float[] convertArrMultiChannelTO1D(float[][][][] inputdata){
        int index=0;
        float[] dest = new float[IMAGE_HEIGHT * IMAGE_WIDTH * OUTPUT_CHANNEL];
        for (int i = 0; i < IMAGE_WIDTH; ++i) {
            for (int j = 0; j < IMAGE_HEIGHT; ++j) {
                index=i* IMAGE_WIDTH +j;
                for (int c=0;c<OUTPUT_CHANNEL;c++){
                    dest[index* OUTPUT_CHANNEL +c]=inputdata[0][i][j][c];
                }
            }
        }
        return dest;
    }

    public void printArr4D(String msg, float[][][][] data, int startRow , int endRow){
        float R,G,B;
        for (int i = startRow; i < endRow; ++i) {
            for (int j = 0; j < IMAGE_HEIGHT; ++j) {
                R =data[0][i][j][0];
                G=data[0][i][j][1];
                B=data[0][i][j][2];
                Log.e("printArr4D,data:",msg+"[" +R+","+G+","+B +"]\n");
            }
        }
    }


    public native void jniImagePro(long matAddrSrcImage, long matAddrDestImage,long A);
}
