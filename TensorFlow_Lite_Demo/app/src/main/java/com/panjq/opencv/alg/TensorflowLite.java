package com.panjq.opencv.alg;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.media.ThumbnailUtils;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class TensorflowLite {
    protected Interpreter tflite;
    static {
        System.loadLibrary("tensorflowlite_jni");
            Log.e("tf","tensorflowlite_jni模型文件加载成功");
    }

    TensorflowLite(Context context, String modePath) {
        //初始化TensorFlowInferenceInterface对象
        tflite = new Interpreter(loadModelFile(context,modePath));
        Log.e("tf","TensoFlow模型文件加载成功");
    }

    /**
     *  加载模型，Memory-map the model file in Assets.
     * @param context
     * @param model_path
     * @return
     */
    private MappedByteBuffer loadModelFile(Context context, String model_path)  {
//        String model_path="mobilenet_quant_v1_224.tflite";
        AssetFileDescriptor fileDescriptor = null;
        try {
            fileDescriptor = context.getAssets().openFd(model_path);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * 给TFlite feed数据，运行推理获得输出结果
     * @param inputs inputs=[batch_size,height,width,channels]
     * @param outputs outputs=[batch_size,D1]
     * @throws IOException
     */
    public void runTFlite(float[][][][] inputs, float[][] outputs) throws IOException{
        tflite.run(inputs, outputs);
    }

    /**
     * 给TFlite feed数据，运行推理获得输出结果
     * @param inputs inputs=[batch_size,height,width,channels]
     * @param outputs outputs=[batch_size,D1,D2]
     * @throws IOException
     */
    public void runTFlite(float[][][][] inputs, float[][][] outputs) throws IOException{
        tflite.run(inputs, outputs);
    }

    /**
     * 给TFlite feed数据，运行推理获得输出结果
     * @param inputs inputs=[batch_size,height,width,channels]
     * @param outputs outputs=[batch_size,height,width,channels] or [batch_size,D1,D2,D3]
     * @throws IOException
     */
    public void runTFlite(float[][][][] inputs, float[][][][] outputs) throws IOException{
        tflite.run(inputs, outputs);
    }


    /**
     * 调用Android Neural Networks API (NNAPI) 进行硬件加速，需要加速芯片支持NNAPI
     * @param nnapi
     */
    public void setUseNNAPI(Boolean nnapi) {
        if (tflite != null)
            tflite.setUseNNAPI(nnapi);
    }

    public void setNumThreads(int num_threads) {
        if (tflite != null)
            tflite.setNumThreads(num_threads);
    }

    /** Closes tflite to release resources. */
    public void close() {
        tflite.close();
        tflite = null;
    }
}
