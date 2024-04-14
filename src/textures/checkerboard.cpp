#include <lightwave.hpp>

namespace lightwave {

class CheckerboardTexture : public Texture {
    Vector2 m_scale;
    Color m_color0;
    Color m_color1;

public:
    CheckerboardTexture(const Properties &properties) {
        m_scale = properties.get<Vector2>("scale");
        m_color0 = properties.get<Color>("color0", Color(0));
        m_color1 = properties.get<Color>("color1", Color(1));
    }

    Color evaluate(const Point2 &uv) const override {
        Point2 scaled_uv = Point2(uv.x() * m_scale.x(), uv.y() * m_scale.y());
        int checker = (static_cast<int>(floor(scaled_uv.x())) + static_cast<int>(floor(scaled_uv.y()))) % 2;
        return (checker == 0) ? m_color0 : m_color1;
    }

    std::string toString() const override {
        return tfm::format("Checkerboard");
    }
};

} // namespace lightwave

REGISTER_TEXTURE(CheckerboardTexture, "checkerboard")
