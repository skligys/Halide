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
#include "edge_detect.h"
#include "preview.h"
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

extern "C" {

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

    buffer_t src1Luma = src1->luma();
    buffer_t src2Luma = src2->luma();
    buffer_t dstLuma = dst->luma();
    buffer_t dstChromaU = dst->chromaU();
    buffer_t dstChromaV = dst->chromaV();

    jfloat *force = env->GetFloatArrayElements(dstForce, 0);

    buffer_t forceBuffer;
    forceBuffer = { 0 };
    forceBuffer.host = reinterpret_cast<uint8_t *>(force);
    forceBuffer.host_dirty = true;
    forceBuffer.extent[0] = 2;
    forceBuffer.stride[0] = 1;
    forceBuffer.elem_size = 4;

    int64_t t1 = halide_current_time_ns();
    int err = edge_detect(&src1Luma, &src2Luma, ballX, ballY, &dstLuma, &dstChromaU, &dstChromaV, &forceBuffer);
    if (err != halide_error_code_success) {
        LOGE("edge_detect() failed with error code: %d", err);
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

    return (err != halide_error_code_success);
}

JNIEXPORT bool JNICALL Java_com_example_helloandroidcamera2_HalideFilters_previewHalide(
    JNIEnv *env, jobject obj, jlong src1YuvBufferTHandle, jlong src2YuvBufferTHandle, jint ballX, jint ballY, jlong dstYuvBufferTHandle, jfloatArray dstForce) {
    if (src1YuvBufferTHandle == 0L || src2YuvBufferTHandle == 0L || dstYuvBufferTHandle == 0L ) {
        LOGE("previewHalide failed: src1, src2 and dst must not be null");
        return false;
    }

    YuvBufferT *src1 = reinterpret_cast<YuvBufferT *>(src1YuvBufferTHandle);
    YuvBufferT *src2 = reinterpret_cast<YuvBufferT *>(src2YuvBufferTHandle);
    YuvBufferT *dst = reinterpret_cast<YuvBufferT *>(dstYuvBufferTHandle);

    if (!equalExtents(*src1, *dst) || !equalExtents(*src2, *dst)) {
        LOGE("previewHalide failed: src and dst extents must be equal.\n\t"
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
        first_call = false;
        for (int t = 0; t < 16; t++) {
            times[t] = 0;
        }
    }

    buffer_t src1Luma = src1->luma();
    buffer_t src2Luma = src2->luma();
    buffer_t src2ChromaU = src2->chromaU();
    buffer_t src2ChromaV = src2->chromaV();
    buffer_t dstLuma = dst->luma();
    buffer_t dstChromaU = dst->chromaU();
    buffer_t dstChromaV = dst->chromaV();

    jfloat *force = env->GetFloatArrayElements(dstForce, 0);

    buffer_t forceBuffer;
    forceBuffer = { 0 };
    forceBuffer.host = reinterpret_cast<uint8_t *>(force);
    forceBuffer.host_dirty = true;
    forceBuffer.extent[0] = 2;
    forceBuffer.stride[0] = 1;
    forceBuffer.elem_size = 4;

    int64_t t1 = halide_current_time_ns();
    int err = preview(&src1Luma, &src2Luma, &src2ChromaU, &src2ChromaV, ballX, ballY, &dstLuma, &dstChromaU, &dstChromaV, &forceBuffer);
    if (err != halide_error_code_success) {
        LOGE("preview() failed with error code: %d", err);
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

    return (err != halide_error_code_success);
}

} // extern "C"
