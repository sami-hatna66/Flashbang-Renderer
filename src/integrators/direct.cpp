#include <lightwave.hpp>

namespace lightwave {

class DirectIntegrator : public SamplingIntegrator {
public:
    DirectIntegrator(const Properties &properties)
    : SamplingIntegrator(properties) {}

    Color Li(const Ray &ray, Sampler &rng) override {
        auto its = m_scene->intersect(ray, rng);
        if (its) {
            if (its.instance->emission()) return its.evaluateEmission();

            // next-event estimation
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
            
            Ray new_ray(its.position, sample_bsdf.wi);
            auto its_2 = m_scene->intersect(new_ray, rng);

            if (its_2) return light_weight + sample_bsdf.weight * its_2.evaluateEmission();
            else return light_weight + sample_bsdf.weight * m_scene->evaluateBackground(new_ray.direction).value;
        } else {
            return m_scene->evaluateBackground(ray.direction).value;
        }
    }

    std::string toString() const override {
        return tfm::format(
            "DirectIntegrator"
        );
    }
};

}

REGISTER_INTEGRATOR(DirectIntegrator, "direct")
