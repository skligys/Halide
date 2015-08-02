package com.example.helloandroidcamera2;

import android.util.Log;

/**
 * Java wrappers for fast filters implemented in Halide.
 */
public class HalideFilters {

    // Load native Halide shared library.
    static {
        System.loadLibrary("native");
    }

    /**
     * A Halide-accelerated diff detector on the luminance channel with force feedback.
     * @return true if it succeeded.
     */
    public static boolean edgeDetect(HalideYuvBufferT src1, HalideYuvBufferT src2, int ballX, int ballY, HalideYuvBufferT dst, float[] force) {
        return HalideFilters.edgeDetectHalide(src1.handle(), src2.handle(), ballX, ballY, dst.handle(), force);
    }

    /**
     * A Halide-accelerated bouncy ball preview with force feedback.
     * @return true if it succeeded.
     */
    public static boolean preview(HalideYuvBufferT src1, HalideYuvBufferT src2, int ballX, int ballY, HalideYuvBufferT dst, float[] force) {
        return HalideFilters.previewHalide(src1.handle(), src2.handle(), ballX, ballY, dst.handle(), force);
    }

    /**
     * A Halide-accelerated bouncy ball with differences displayed. Chroma is set to 128.
     * @eturns true if it succeeded.
     */
    private static native boolean edgeDetectHalide(long src1YuvHandle, long src2YuvHandle, int ballX, int ballY, long dstYuvHandle, float[] force);

    /**
     * A Halide-accelerated bouncy ball preview.
     * @eturns true if it succeeded.
     */
    private static native boolean previewHalide(long src1YuvHandle, long src2YuvHandle, int ballX, int ballY, long dstYuvHandle, float[] force);
}
