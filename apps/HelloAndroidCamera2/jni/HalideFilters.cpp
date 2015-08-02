#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <android/native_window_jni.h>

#include <algorithm>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "native", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "native", __VA_ARGS__)

#include "AndroidBufferUtilities.h"
#include "BufferTFunctions.h"
#include "YuvFunctions.h"
#include "deinterleave.h"
#include "edge_detect.h"
#include "HalideRuntime.h"

#define DEBUG 1

// Extern functions from the Halide runtime that are not exposed in
// HalideRuntime.h.
extern "C" int halide_host_cpu_count();
extern "C" int64_t halide_current_time_ns();

// Override Halide's print to use LOGD and also print the time.
extern "C" void halide_print(void *, const char *msg) {
    static int64_t t0 = halide_current_time_ns();
    int64_t t1 = halide_current_time_ns();
    LOGD("%d: %s\n", (int)(t1 - t0)/1000000, msg);
    t0 = t1;
}

bool addBouncyBall(uint32_t ballX, uint32_t ballY, buffer_t &luma, buffer_t &chromaU, buffer_t &chromaV);

extern "C" {

JNIEXPORT bool JNICALL Java_com_example_helloandroidcamera2_HalideFilters_copyHalide(
    JNIEnv *env, jobject obj, jlong srcYuvBufferTHandle, jint ballX, jint ballY, jlong dstYuvBufferTHandle) {
    if (srcYuvBufferTHandle == 0L || dstYuvBufferTHandle == 0L ) {
        LOGE("copyHalide failed: src and dst must not be null");
        return false;
    }

    YuvBufferT *src = reinterpret_cast<YuvBufferT *>(srcYuvBufferTHandle);
    YuvBufferT *dst = reinterpret_cast<YuvBufferT *>(dstYuvBufferTHandle);

    if (!equalExtents(*src, *dst)) {
        LOGE("copyHalide failed: src and dst extents must be equal.\n\t"
            "src extents: luma: %d, %d, chromaU: %d, %d, chromaV: %d, %d.\n\t"
            "dst extents: luma: %d, %d, chromaU: %d, %d, chromaV: %d, %d.",
            src->luma().extent[0], src->luma().extent[1],
            src->chromaU().extent[0], src->chromaU().extent[1],
            src->chromaV().extent[0], src->chromaV().extent[1],
            dst->luma().extent[0], dst->luma().extent[1],
            dst->chromaU().extent[0], dst->chromaU().extent[1],
            dst->chromaV().extent[0], dst->chromaV().extent[1]);
        return false;
    }

    YuvBufferT::ChromaStorage srcChromaStorage = src->chromaStorage();
    YuvBufferT::ChromaStorage dstChromaStorage = dst->chromaStorage();

    bool succeeded;
    int halideErrorCode;

    // Use Halide deinterleave if the source chroma is interleaved and destination chroma is planar.
    // Other, fall back to slow copy.
    if ((srcChromaStorage == YuvBufferT::ChromaStorage::kInterleavedUFirst ||
         srcChromaStorage == YuvBufferT::ChromaStorage::kInterleavedVFirst) &&
         (dstChromaStorage == YuvBufferT::ChromaStorage::kPlanarPackedUFirst ||
          dstChromaStorage == YuvBufferT::ChromaStorage::kPlanarPackedVFirst ||
          dstChromaStorage == YuvBufferT::ChromaStorage::kPlanarGeneric)) {

        // SK: Don't handle this case since it doesn't apply to MotoX 2014.

        // Always copy the luma channel directly, potentially falling back to something slow.
        succeeded = copy2D(src->luma(), dst->luma());
        if (succeeded) {
            // Use Halide to deinterleave the chroma channels.
            buffer_t srcInterleavedChroma = src->interleavedChromaView();
            buffer_t dstPlanarChromaU = dst->chromaU();
            buffer_t dstPlanarChromaV = dst->chromaV();
            if (srcChromaStorage == YuvBufferT::ChromaStorage::kInterleavedUFirst) {
                halideErrorCode = deinterleave(&srcInterleavedChroma,
                    &dstPlanarChromaU, &dstPlanarChromaV);
            } else {
                halideErrorCode = deinterleave(&srcInterleavedChroma,
                    &dstPlanarChromaV, &dstPlanarChromaU);
            }
            succeeded = (halideErrorCode != halide_error_code_success);
            if (halideErrorCode != halide_error_code_success) {
                LOGE("deinterleave failed with error code: %d", halideErrorCode);
            }
        }
    } else {
        succeeded = flipHorizontal2D(src->luma(), dst->luma());
        if (succeeded) {
            succeeded = flipHorizontal2D(src->chromaU(), dst->chromaU());
        }
        if (succeeded) {
            succeeded = flipHorizontal2D(src->chromaV(), dst->chromaV());
        }
        if (succeeded) {
            buffer_t luma = dst->luma();
            buffer_t chromaU = dst->chromaU();
            buffer_t chromaV = dst->chromaV();
            succeeded = addBouncyBall(ballX, ballY, luma, chromaU, chromaV);
        }
    }

    return succeeded;
}

JNIEXPORT bool JNICALL Java_com_example_helloandroidcamera2_HalideFilters_edgeDetectHalide(
    JNIEnv *env, jobject obj, jlong src1YuvBufferTHandle, jlong src2YuvBufferTHandle, jint ballX, jint ballY, jlong dstYuvBufferTHandle, jfloatArray dstForce) {
    if (src1YuvBufferTHandle == 0L || src2YuvBufferTHandle == 0L || dstYuvBufferTHandle == 0L ) {
        LOGE("edgeDetectHalide failed: src1, src2 and dst must not be null");
        return false;
    }

    YuvBufferT *src1 = reinterpret_cast<YuvBufferT *>(src1YuvBufferTHandle);
    YuvBufferT *src2 = reinterpret_cast<YuvBufferT *>(src2YuvBufferTHandle);
    YuvBufferT *dst = reinterpret_cast<YuvBufferT *>(dstYuvBufferTHandle);

    if (!equalExtents(*src1, *dst) || !equalExtents(*src2, *dst)) {
        LOGE("edgeDetectHalide failed: src and dst extents must be equal.\n\t"
            "src1 extents: luma: %d, %d, chromaU: %d, %d, chromaV: %d, %d.\n\t"
            "src2 extents: luma: %d, %d, chromaU: %d, %d, chromaV: %d, %d.\n\t"
            "dst extents: luma: %d, %d, chromaU: %d, %d, chromaV: %d, %d.",
            src1->luma().extent[0], src1->luma().extent[1],
            src1->chromaU().extent[0], src1->chromaU().extent[1],
            src1->chromaV().extent[0], src1->chromaV().extent[1],
            src2->luma().extent[0], src2->luma().extent[1],
            src2->chromaU().extent[0], src2->chromaU().extent[1],
            src2->chromaV().extent[0], src2->chromaV().extent[1],
            dst->luma().extent[0], dst->luma().extent[1],
            dst->chromaU().extent[0], dst->chromaU().extent[1],
            dst->chromaV().extent[0], dst->chromaV().extent[1]);
        return false;
    }

    static bool first_call = true;
    static unsigned counter = 0;
    static unsigned times[16];
    if (first_call) {
        LOGD("According to Halide, host system has %d cpus\n",
             halide_host_cpu_count());
        first_call = false;
        for (int t = 0; t < 16; t++) {
            times[t] = 0;
        }
    }

    // Set chrominance to 128 to appear grayscale.
    if (dst->interleavedChromaView().host != nullptr) {
        fill2D(dst->interleavedChromaView(), 128);
    } else if (dst->packedPlanarChromaView().host != nullptr) {
        fill2D(dst->packedPlanarChromaView(), 128);
    } else {
        fill2D(dst->chromaU(), 128);
        fill2D(dst->chromaV(), 128);
    }

    buffer_t src1Luma = src1->luma();
    buffer_t src2Luma = src2->luma();
    buffer_t dstLuma = dst->luma();

    jfloat *force = env->GetFloatArrayElements(dstForce, 0);

    buffer_t forceBuffer;
    forceBuffer = { 0 };
    forceBuffer.host = reinterpret_cast<uint8_t *>(force);
    forceBuffer.host_dirty = true;
    forceBuffer.extent[0] = 2;
    forceBuffer.stride[0] = 1;
    forceBuffer.elem_size = 4;

    int64_t t1 = halide_current_time_ns();
    int err = edge_detect(&src1Luma, &src2Luma, ballX, ballY, &dstLuma, &forceBuffer);
    if (err != halide_error_code_success) {
        LOGE("edge_detect failed with error code: %d", err);
    }

    int64_t t2 = halide_current_time_ns();
    unsigned elapsed_us = (t2 - t1) / 1000;

    times[counter & 15] = elapsed_us;
    counter++;
    unsigned min = times[0];
    for (int i = 1; i < 16; i++) {
        if (times[i] < min) {
            min = times[i];
        }
    }
    LOGD("Time taken: %d us (minimum: %d us)", elapsed_us, min);

    env->ReleaseFloatArrayElements(dstForce, force, 0);

    if (err == halide_error_code_success) {
        buffer_t luma = dst->luma();
        buffer_t chromaU = dst->chromaU();
        buffer_t chromaV = dst->chromaV();
        bool succeeded = addBouncyBall(ballX, ballY, luma, chromaU, chromaV);
        if (!succeeded) {
            err = halide_error_code_internal_error;
        }
    }

    return (err != halide_error_code_success);
}

} // extern "C"

bool addBouncyBall(uint32_t ballX, uint32_t ballY, buffer_t &luma, buffer_t &chromaU, buffer_t &chromaV) {
    // Fixed size 32x32, needs to match Camera2BasicFragment.Ball.SIZE and EdgeDetect.ball_size.
    const uint32_t SIZE = 32;

    // Check the bounds.
    if (ballX + SIZE >= luma.extent[0] || ballY + SIZE >= luma.extent[1]) {
        LOGE("addBouncyBall() out of bounds:\n\t"
            "ball position: (%d, %d), ball size: (%d, %d), frame size: (%d, %d).",
            ballX, ballY, SIZE, SIZE, luma.extent[0], luma.extent[1]);
        return false;
    }

    // Only do byte sized components, that's all we need for the demo.
    if (luma.elem_size != 1 || chromaU.elem_size != 1 || chromaV.elem_size != 1) {
        LOGE("addBouncyBall() unsupported component size:\n\t"
            "luma.elem_size: %d, chromaU.elem_size: %d, chromaV.elem_size: %d.",
            luma.elem_size, chromaU.elem_size, chromaV.elem_size);
    }

    int32_t lumaElementStrideBytes = luma.stride[0] * luma.elem_size;
    int32_t lumaRowStrideBytes = luma.stride[1] * luma.elem_size;
    int32_t chromaURowStrideBytes = chromaU.stride[1] * chromaU.elem_size;
    int32_t chromaUElementStrideBytes = chromaU.stride[0] * chromaU.elem_size;
    int32_t chromaVRowStrideBytes = chromaV.stride[1] * chromaV.elem_size;
    int32_t chromaVElementStrideBytes = chromaV.stride[0] * chromaV.elem_size;
    for (int y = ballY; y < ballY + SIZE; ++y) {
        uint8_t *lumaRow = luma.host + y * lumaRowStrideBytes;
        uint8_t *chromaURow = chromaU.host + y / 2 * chromaURowStrideBytes;
        uint8_t *chromaVRow = chromaV.host + y / 2 * chromaVRowStrideBytes;
        for (int x = ballX; x < ballX + SIZE; ++x) {
            // Bright green: YUV = (255, 0, 0).
            lumaRow[x * lumaElementStrideBytes] = 255;
            chromaURow[x / 2 * chromaUElementStrideBytes] = 0;
            chromaVRow[x / 2 * chromaVElementStrideBytes] = 0;
        }
    }
}
