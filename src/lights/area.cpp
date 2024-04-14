#include <lightwave.hpp>

namespace lightwave {

/**
 * Sample a light at each intersection, in proportion to the light's size
*/
class AreaLight final : public Light {
private:
    ref<Instance> m_instance;
public:
    AreaLight(const Properties &properties) {
        m_instance = properties.getChild<Instance>();
    }

    DirectLightSample sampleDirect(const Point &origin,
                                   Sampler &rng) const override {
        DirectLightSample result;

        AreaSample area_sample = m_instance->sampleArea(rng);
        result.wi = area_sample.position - origin;
        result.distance = result.wi.length();
        result.wi = result.wi.normalized();
        float pdf = area_sample.pdf * (sqr(result.distance) / abs(area_sample.frame.normal.dot(result.wi)));

        result.weight = m_instance->emission()->evaluate(area_sample.uv, area_sample.frame.toLocal(result.wi)).value / pdf;

        return result;
    }

    bool canBeIntersected() const override { return false; }

    std::string toString() const override {
        return tfm::format("AreaLight[\n"
                           "]");
    }
};

} // namespace lightwave

REGISTER_LIGHT(AreaLight, "area")
