#include <lightwave.hpp>

namespace lightwave {

class Diffuse : public Bsdf {
    ref<Texture> m_albedo;
public:
    Diffuse(const Properties &properties) {
        m_albedo = properties.get<Texture>("albedo");
    }

    Color albedo(const Point2 &uv) override {
        return m_albedo->evaluate(uv);
    }

    BsdfEval evaluate(const Point2 &uv, const Vector &wo,
                      const Vector &wi) const override {
        BsdfEval result;

        result.value = (Frame::cosTheta(wi) * m_albedo->evaluate(uv)) / Pi;

        return result;
    }

    BsdfSample sample(const Point2 &uv, const Vector &wo,
                      Sampler &rng) const override {
        BsdfSample result;

        result.wi = squareToCosineHemisphere(rng.next2D());

        if (cosineHemispherePdf(result.wi) != 0) {
            result.weight = (Frame::cosTheta(result.wi) * m_albedo->evaluate(uv)) / (cosineHemispherePdf(result.wi) * Pi);
        } else {
            result.weight = Color(0);
        }
        
        return result;
    }

    std::string toString() const override {
        return tfm::format("Diffuse[\n"
                           "  albedo = %s\n"
                           "]",
                           indent(m_albedo));
    }
};

} // namespace lightwave

REGISTER_BSDF(Diffuse, "diffuse")
