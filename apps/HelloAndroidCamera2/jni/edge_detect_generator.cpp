#include "Halide.h"

namespace {

class EdgeDetect : public Halide::Generator<EdgeDetect> {
public:
    ImageParam input1{ UInt(8), 2, "input1" };
    ImageParam input2{ UInt(8), 2, "input2" };
    Param<uint32_t> ball_x{ "ball_x", 0 };
    Param<uint32_t> ball_y{ "ball_y", 0 };
    // Important: Needs to match HalidFilters::addBouncyBall::SIZE and Camera2BasicFragment.Ball.SIZE.
    GeneratorParam<int> ball_size{ "ball_size", 32 };
    GeneratorParam<float> force_factor{ "force_factor", 10.0 };

    Pipeline build() {
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
        clamped_abs_diff(x, y) = clamp(abs_diff(x, y), 5, 255) - 5;

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

        // Ball area.
        RDom r(ball_x, ball_size, ball_y, ball_size);
        // Force a pixel imparts on the ball is proportional to result and distance from the ball's center.
        Func norm_result;
        norm_result(x, y) = cast<float>(flipped_result(x, y)) / 255.0f;
        const int ball_size_half = ball_size / 2;
        Func pixel_force_x;
        pixel_force_x(x, y) = force_factor * norm_result(x, y) * cast<float>((ball_x + ball_size_half) - x);
        Func pixel_force_y;
        pixel_force_y(x, y) = force_factor * norm_result(x, y) * cast<float>((ball_y + ball_size_half) - y);

        Func force_x;
        Func force_y;
        force_x() = 0.0f;
        force_y() = 0.0f;
        force_x() += pixel_force_x(r.x, r.y);
        force_y() += pixel_force_y(r.x, r.y);
        Func force;
        force(x) = 0.0f;
        force(0) = force_x();
        force(1) = force_y();
        force.compute_root();

        return Pipeline({flipped_result, force});
    }
};

Halide::RegisterGenerator<EdgeDetect> register_edge_detect{ "edge_detect" };

}  // namespace
