#include <lightwave.hpp>

namespace lightwave {

/**
 * @brief A perspective camera with a given field of view angle and transform.
 * 
 * In local coordinates (before applying m_transform), the camera looks in positive z direction [0,0,1].
 * Pixels on the left side of the image ( @code normalized.x < 0 @endcode ) are directed in negative x
 * direction ( @code ray.direction.x < 0 ), and pixels at the bottom of the image ( @code normalized.y < 0 @endcode )
 * are directed in negative y direction ( @code ray.direction.y < 0 ).
 */
class Perspective : public Camera {
private:
    float m_scale;
    enum class FovAxis {x, y};
    FovAxis m_fov_axis;
    float m_aspect_ratio;
public:
    Perspective(const Properties &properties)
    : Camera(properties) {
        // scale to fov angle
        float fov = properties.get<float>("fov");
        m_scale = tan(fov * 0.5f * Deg2Rad);

        // calculate aspect ratio (fovAxis dependent)
        auto fov_axis = properties.get<std::string>("fovAxis");
        m_fov_axis = fov_axis == "x" ? FovAxis::x : FovAxis::y;
        if (m_fov_axis == FovAxis::x) m_aspect_ratio = (float) m_resolution.y() / (float) m_resolution.x();
        else m_aspect_ratio = (float) m_resolution.x() / (float) m_resolution.y();
    }

    CameraSample sample(const Point2 &normalized, Sampler &rng) const override {
        // Focal vector points at image plane, unit length
        Vector f{0.f, 0.f, 1.f};

        // Both components get scaled by m_scale (accounts for fov)
        // Component that isn't fovAxis gets scaled by aspect ratio
        float p_x = normalized.x() * m_scale * (m_fov_axis == FovAxis::y ? m_aspect_ratio : 1.f);
        float p_y = normalized.y() * m_scale * (m_fov_axis == FovAxis::x ? m_aspect_ratio : 1.f);
    
        Vector s_x{p_x, 0.f, 0.f};
        Vector s_y{0.f, p_y, 0.f};
 
        Vector d = f + s_x + s_y;

        return CameraSample{
            // Map local to world
            .ray = m_transform->apply(Ray(Vector(0.f, 0.f, 0.f), d)).normalized(),
            .weight = Color(1.f)
        };
    }

    std::string toString() const override {
        return tfm::format(
            "Perspective[\n"
            "  width = %d,\n"
            "  height = %d,\n"
            "  transform = %s,\n"
            "]",
            m_resolution.x(),
            m_resolution.y(),
            indent(m_transform)
        );
    }
};

}

REGISTER_CAMERA(Perspective, "perspective")
