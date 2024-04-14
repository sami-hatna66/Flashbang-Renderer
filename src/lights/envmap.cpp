#include <lightwave.hpp>

namespace lightwave {

class EnvironmentMap final : public BackgroundLight {
    /// @brief The texture to use as background
    ref<Texture> m_texture;

    /// @brief An optional transform from local-to-world space
    ref<Transform> m_transform;

public:
    EnvironmentMap(const Properties &properties) {
        m_texture   = properties.getChild<Texture>();
        m_transform = properties.getOptionalChild<Transform>();
    }

    BackgroundLightEval evaluate(const Vector &direction) const override {
        Vector2 warped = Vector2(0, 0);

        auto trans_dir = direction;
        if (m_transform) {
            trans_dir = m_transform->inverse(direction);
        }

        float theta = acos(trans_dir.y());
        float phi = atan2(trans_dir.z(), trans_dir.x());
        warped.x() = 0.5 - (0.5 * InvPi * (phi));
        warped.y() = theta * InvPi;

        return {
            .value = m_texture->evaluate(warped),
        };
    }

    DirectLightSample sampleDirect(const Point &origin,
                                   Sampler &rng) const override {
        Vector direction = squareToUniformSphere(rng.next2D());
        auto E           = evaluate(direction);

        return {
            .wi     = direction,
            .weight = E.value / Inv4Pi,
            .distance = Infinity,
        };
    }

    std::string toString() const override {
        return tfm::format("EnvironmentMap[\n"
                           "  texture = %s,\n"
                           "  transform = %s\n"
                           "]",
                           indent(m_texture), indent(m_transform));
    }
};

} // namespace lightwave

REGISTER_LIGHT(EnvironmentMap, "envmap")
