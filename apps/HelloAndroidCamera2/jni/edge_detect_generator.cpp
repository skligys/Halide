#include "Halide.h"

namespace {

class EdgeDetect : public Halide::Generator<EdgeDetect> {
public:
    ImageParam input1{ UInt(8), 2, "input1" };
    ImageParam input2{ UInt(8), 2, "input2" };

    Func build() {
        Var x, y;

        // Upcast to 16-bit
        Func input1_16;
        input1_16(x, y) = cast<int16_t>(input1(x, y));
        Func input2_16;
        input2_16(x, y) = cast<int16_t>(input2(x, y));

        // Absolute difference.
        Func diff;
        diff(x, y) = input2_16(x, y) - input1_16(x, y);
        Func abs_diff;
        abs_diff(x, y) = Halide::abs(diff(x, y));

        // Draw the result
        Func result;
        result(x, y) = cast<uint8_t>(clamp(abs_diff(x, y), 0, 255));

        // CPU schedule:
        //   Parallelize over scan lines, 4 scanlines per task.
        //   Independently, vectorize in x.
        result
            .compute_root()
            .vectorize(x, 8)
            .parallel(y, 8);

        return result;
    }
};

Halide::RegisterGenerator<EdgeDetect> register_edge_detect{ "edge_detect" };

}  // namespace
