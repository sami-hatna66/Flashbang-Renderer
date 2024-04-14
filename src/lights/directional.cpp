#include <lightwave.hpp>

namespace lightwave {

class DirectionalLight final : public Light {
private:
    Vector m_direction;
    Color m_intensity;
public:
    DirectionalLight(const Properties &properties) {
        m_direction = properties.get<Vector>("direction").normalized();
        m_intensity = properties.get<Color>("intensity");
    }

    DirectLightSample sampleDirect(const Point &origin,
                                   Sampler &rng) const override {
        DirectLightSample result;

        result.wi = m_direction;
        result.distance = Infinity;
        result.weight = m_intensity;

        return result;
    }

    bool canBeIntersected() const override { return false; }

    std::string toString() const override {
        return tfm::format("DirectionalLight[\n"
                           "]");
    }
};

} // namespace lightwave

REGISTER_LIGHT(DirectionalLight, "directional")
