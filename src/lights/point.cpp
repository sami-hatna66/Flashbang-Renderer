#include <lightwave.hpp>

namespace lightwave {

class PointLight final : public Light {
private:
    Point m_position;
    Color m_power;
public:
    PointLight(const Properties &properties) {
        m_position = properties.get<Point>("position");
        m_power = properties.get<Color>("power");
    }

    DirectLightSample sampleDirect(const Point &origin,
                                   Sampler &rng) const override {
        DirectLightSample result;
        
        result.wi = m_position - origin;
        result.distance = result.wi.length();
        result.wi = result.wi.normalized();
        result.weight = m_power / (4.f * Pi * result.distance * result.distance);

        return result;
    }

    bool canBeIntersected() const override { return false; }

    std::string toString() const override {
        return tfm::format("PointLight[\n"
                           "]");
    }
};

} // namespace lightwave

REGISTER_LIGHT(PointLight, "point")
