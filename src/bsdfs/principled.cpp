#include <lightwave.hpp>

#include "fresnel.hpp"
#include "microfacet.hpp"

namespace lightwave {

struct DiffuseLobe {
    Color color;

    BsdfEval evaluate(const Vector &wo, const Vector &wi) const {
        BsdfEval result;

        result.value = (wi.z() * color) / Pi;

        return result;
    }

    BsdfSample sample(const Vector &wo, Sampler &rng) const {
        BsdfSample result;

        result.wi = squareToCosineHemisphere(rng.next2D());

        if (cosineHemispherePdf(result.wi) != 0) {
            result.weight = (result.wi.z() * color) / (cosineHemispherePdf(result.wi) * Pi);
        } else {
            result.weight = Color(0);
        }
        
        return result;
    }
};

struct MetallicLobe {
    float alpha;
    Color color;

    BsdfEval evaluate(const Vector &wo, const Vector &wi) const {
        BsdfEval result;

        auto wm = (wi + wo).normalized();
        auto R = color;
        auto D = microfacet::evaluateGGX(alpha, wm);
        auto g1wi = microfacet::smithG1(alpha, wm, wi);
        auto g1wo = microfacet::smithG1(alpha, wm, wo);

        result.value = ((R * D * g1wi * g1wo) / abs(4.f * Frame::cosTheta(wo)));

        return result;
    }

    BsdfSample sample(const Vector &wo, Sampler &rng) const {
        BsdfSample result;

        auto wm = microfacet::sampleGGXVNDF(alpha, wo, rng.next2D());
        result.wi = reflect(wo, wm);

        auto R = color;
        auto g1wi = microfacet::smithG1(alpha, wm, result.wi);

        result.weight = R * g1wi;

        return result;
    }
};

class Principled : public Bsdf {
    ref<Texture> m_baseColor;
    ref<Texture> m_roughness;
    ref<Texture> m_metallic;
    ref<Texture> m_specular;

    struct Combination {
        float diffuseSelectionProb;
        DiffuseLobe diffuse;
        MetallicLobe metallic;
    };

    Combination combine(const Point2 &uv, const Vector &wo) const {
        const auto baseColor = m_baseColor->evaluate(uv);
        const auto alpha = std::max(float(1e-3), sqr(m_roughness->scalar(uv)));
        const auto specular = m_specular->scalar(uv);
        const auto metallic = m_metallic->scalar(uv);
        const auto F =
            specular * schlick((1 - metallic) * 0.08f, Frame::cosTheta(wo));

        const DiffuseLobe diffuseLobe = {
            .color = (1 - F) * (1 - metallic) * baseColor,
        };
        const MetallicLobe metallicLobe = {
            .alpha = alpha,
            .color = F * Color(1) + (1 - F) * metallic * baseColor,
        };

        const auto diffuseAlbedo = diffuseLobe.color.mean();
        const auto totalAlbedo =
            diffuseLobe.color.mean() + metallicLobe.color.mean();
        return {
            .diffuseSelectionProb =
                totalAlbedo > 0 ? diffuseAlbedo / totalAlbedo : 1.0f,
            .diffuse  = diffuseLobe,
            .metallic = metallicLobe,
        };
    }

public:
    Principled(const Properties &properties) {
        m_baseColor = properties.get<Texture>("baseColor");
        m_roughness = properties.get<Texture>("roughness");
        m_metallic  = properties.get<Texture>("metallic");
        m_specular  = properties.get<Texture>("specular");
    }

    Color albedo(const Point2 &uv) override {
        return m_baseColor->evaluate(uv);
    }

    BsdfEval evaluate(const Point2 &uv, const Vector &wo,
                      const Vector &wi) const override {
        const auto combination = combine(uv, wo);
        
        auto eval_diffuse = combination.diffuse.evaluate(wo, wi);
        auto eval_metallic = combination.metallic.evaluate(wo, wi);

        BsdfEval result;
        result.value = eval_diffuse.value + eval_metallic.value;
        return result;
    }

    BsdfSample sample(const Point2 &uv, const Vector &wo,
                      Sampler &rng) const override {
        const auto combination = combine(uv, wo);
        if (rng.next() < combination.diffuseSelectionProb) {
            // diffuse
            auto sample_diffuse = combination.diffuse.sample(wo, rng);
            sample_diffuse.weight /= combination.diffuseSelectionProb;
            return sample_diffuse;
        } else {
            // metallic
            auto sample_metallic = combination.metallic.sample(wo, rng);
            sample_metallic.weight /= (1.f - combination.diffuseSelectionProb);
            return sample_metallic;
        }
    }

    std::string toString() const override {
        return tfm::format("Principled[\n"
                           "  baseColor = %s,\n"
                           "  roughness = %s,\n"
                           "  metallic  = %s,\n"
                           "  specular  = %s,\n"
                           "]",
                           indent(m_baseColor), indent(m_roughness),
                           indent(m_metallic), indent(m_specular));
    }
};

} // namespace lightwave

REGISTER_BSDF(Principled, "principled")
