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
     * Copy one Yuv image to another. They must have the same size (but can
     * have different strides). Uses Halide for fast UV deinterleaving if
     * formats are compatible.
     * @return true if it succeeded.
     */
    public static boolean copy(HalideYuvBufferT src, int ballX, int ballY, HalideYuvBufferT dst) {
        return HalideFilters.copyHalide(src.handle(), ballX, ballY, dst.handle());
    }

    /**
     * A Halide-accelerated edge detector on the luminance channel.
     * @return true if it succeeded.
     */
    public static boolean edgeDetect(HalideYuvBufferT src1, HalideYuvBufferT src2, int ballX, int ballY, HalideYuvBufferT dst) {
        return HalideFilters.edgeDetectHalide(src1.handle(), src2.handle(), ballX, ballY, dst.handle());
    }

    /**
     * A Halide-accelerated native copy between two native Yuv handles.
     * @return true if it succeeded.
     */
    private static native boolean copyHalide(long srcYuvHandle, int ballX, int ballY, long dstYuvHandle);

    /**
     * A Halide-accelerated edge detector on the luminance channel. Chroma is set to 128.
     * @eturns true if it succeeded.
     */
    private static native boolean edgeDetectHalide(long src1YuvHandle, long src2YuvHandle, int ballX, int ballY, long dstYuvHandle);
}
