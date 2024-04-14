#include <lightwave.hpp>

namespace lightwave {

class PathTracer : public SamplingIntegrator {
private:
    int m_depth;
public:
    PathTracer(const Properties &properties)
    : SamplingIntegrator(properties) {
        m_depth = properties.get<int>("depth", 2);
    }

    Color Li(const Ray &ray, Sampler &rng) override {
        auto its = m_scene->intersect(ray, rng);
        if (its) {
            if (its.instance->emission()) return its.evaluateEmission();

            if (ray.depth >= m_depth - 1) return Color(0);

            Color light_weight(0.f);
            if (m_scene->hasLights()) {
                auto light = m_scene->sampleLight(rng);
                if (!light.light->canBeIntersected()) {
                    auto light_sample = light.light->sampleDirect(its.position, rng);
                    auto intersect_shadow = m_scene->intersect(Ray(its.position, light_sample.wi), rng);
                    if (!intersect_shadow || 
                        (intersect_shadow.position - its.position).length() >= light_sample.distance) {
                        auto bsdf_eval = its.evaluateBsdf(light_sample.wi).value;
                        light_weight = max(Color(0), bsdf_eval * (light_sample.weight / light.probability));
                    }
                }
            }

            auto sample_bsdf = its.sampleBsdf(rng);
            Ray new_ray = Ray(its.position, sample_bsdf.wi);
            new_ray.depth = ray.depth + 1;

            if (new_ray.depth < m_depth) {
                return light_weight + sample_bsdf.weight * Li(new_ray, rng);
            } else {
                return light_weight + sample_bsdf.weight * m_scene->evaluateBackground(new_ray.direction).value;
            }
        } else {
            return m_scene->evaluateBackground(ray.direction).value;
        }
    }

    std::string toString() const override {
        return tfm::format(
            "PathTracer"
        );
    }
};

}

REGISTER_INTEGRATOR(PathTracer, "pathtracer")
