#include "Halide.h"

namespace {

class EdgeDetect : public Halide::Generator<EdgeDetect> {
public:
    ImageParam luma1{ UInt(8), 2, "luma1" };
    ImageParam luma2{ UInt(8), 2, "luma2" };
    Param<uint32_t> ball_x{ "ball_x", 0 };
    Param<uint32_t> ball_y{ "ball_y", 0 };
    // Important: Needs to match HalidFilters::addBouncyBall::SIZE and Camera2BasicFragment.Ball.SIZE.
    GeneratorParam<int> ball_size{ "ball_size", 64 };
    GeneratorParam<float> force_factor{ "force_factor", 10.0 };

    Pipeline build() {
        Var x, y;

        // Upcast to 16-bit.
        Func luma1_16;
        luma1_16(x, y) = cast<int16_t>(luma1(x, y));
        Func luma2_16;
        luma2_16(x, y) = cast<int16_t>(luma2(x, y));

        // Absolute pixel by pixel difference.
        Func abs_diff;
        abs_diff(x, y) = Halide::abs(luma2_16(x, y) - luma1_16(x, y));

        // Clamp, eliminate sensor noise.
        Func clamped_abs_diff;
        clamped_abs_diff(x, y) = clamp(abs_diff(x, y), 5, 255) - 5;

        // Draw the result, flip horizontally to account for front facing camera's orientation.
        Func result_luma;
        result_luma(x, y) = cast<uint8_t>(clamped_abs_diff(x, y));
        Func flipped_result_luma;
        flipped_result_luma(x, y) = result_luma(luma1.width() - 1 - x, y);

        // CPU schedule:
        //   Parallelize over scan lines, 8 scanlines per task.
        //   Independently, vectorize in x.
        flipped_result_luma
            .compute_root()
            .vectorize(x, 8)
            .parallel(y, 8);

        // Result chroma: (128, 128) almost everywhere to make diffs grayscale.  (0, 0) inside the ball to make it green.
        Func result_chroma_u;
        Func result_chroma_v;
        result_chroma_u(x, y) = cast<uint8_t>(128);
        result_chroma_v(x, y) = cast<uint8_t>(128);

        // Ball area.
        RDom r(ball_x, ball_size, ball_y, ball_size);

        // Ball area in chroma plane, image 2 times smaller.
        RDom r_uv(ball_x / 2, ball_size / 2, ball_y / 2, ball_size / 2);
        result_chroma_u(r_uv.x, r_uv.y) = cast<uint8_t>(0);
        result_chroma_v(r_uv.x, r_uv.y) = cast<uint8_t>(0);
        result_chroma_u
            .compute_root()
            .vectorize(x, 8)
            .parallel(y, 8);
        result_chroma_v
            .compute_root()
            .vectorize(x, 8)
            .parallel(y, 8);

        // Force a pixel imparts on the ball is proportional to result and distance from the ball's center.
        Func norm_result_luma;
        norm_result_luma(x, y) = cast<float>(flipped_result_luma(x, y)) / 255.0f;
        const int ball_size_half = ball_size / 2;
        Func pixel_force_x;
        pixel_force_x(x, y) = force_factor * norm_result_luma(x, y) * cast<float>((ball_x + ball_size_half) - x);
        Func pixel_force_y;
        pixel_force_y(x, y) = force_factor * norm_result_luma(x, y) * cast<float>((ball_y + ball_size_half) - y);

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

        return Pipeline({flipped_result_luma, result_chroma_u, result_chroma_v, force});
    }
};

Halide::RegisterGenerator<EdgeDetect> register_edge_detect{ "edge_detect" };

}  // namespace
