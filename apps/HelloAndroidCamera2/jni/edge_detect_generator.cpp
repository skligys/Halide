#include "Halide.h"

namespace {

class EdgeDetect : public Halide::Generator<EdgeDetect> {
public:
    ImageParam input1{ UInt(8), 2, "input1" };
    ImageParam input2{ UInt(8), 2, "input2" };

    Func build() {
        Var x, y;

        // Upcast to 16-bit.
        Func input1_16;
        input1_16(x, y) = cast<int16_t>(input1(x, y));
        Func input2_16;
        input2_16(x, y) = cast<int16_t>(input2(x, y));

        // Absolute pixel by pixel difference.
        Func abs_diff;
        abs_diff(x, y) = Halide::abs(input2_16(x, y) - input1_16(x, y));

        // Clamp, eliminate sensor noise.
        Func clamped_abs_diff;
        clamped_abs_diff(x, y) = clamp(abs_diff(x, y), 10, 255) - 10;

        // Draw the result, flip horizontally to account for front facing camera's orientation.
        Func result;
        result(x, y) = cast<uint8_t>(clamped_abs_diff(x, y));
        Func flipped_result;
        flipped_result(x, y) = result(input1.width() - 1 - x, y);

        // CPU schedule:
        //   Parallelize over scan lines, 8 scanlines per task.
        //   Independently, vectorize in x.
        flipped_result
            .compute_root()
            .vectorize(x, 8)
            .parallel(y, 8);

        return flipped_result;
    }
};

Halide::RegisterGenerator<EdgeDetect> register_edge_detect{ "edge_detect" };

}  // namespace
