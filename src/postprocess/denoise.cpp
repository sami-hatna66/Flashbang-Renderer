#ifdef LW_WITH_OIDN

#include <lightwave.hpp>
#include <OpenImageDenoise/oidn.hpp>

namespace lightwave {

class Denoise : public Postprocess {
private:
    ref<Image> m_normals;
    ref<Image> m_albedo;

public:
    Denoise(const Properties &properties) : Postprocess(properties) {
        m_normals = properties.get<Image>("normals");
        m_albedo = properties.get<Image>("albedo");
    }

    void execute() override {
        const Point2i resolution = m_input->resolution();
        m_output->initialize(resolution);

        oidn::DeviceRef device = oidn::newDevice();
        device.commit();

        oidn::BufferRef color_buf = device.newBuffer(m_input->data(), resolution.x() * resolution.y() * sizeof(Color));
        oidn::BufferRef normals_buf = device.newBuffer(m_normals->data(), resolution.x() * resolution.y() * sizeof(Color));
        oidn::BufferRef albedo_buf = device.newBuffer(m_albedo->data(), resolution.x() * resolution.y() * sizeof(Color));

        oidn::FilterRef filter = device.newFilter("RT");
        filter.setImage("color", color_buf, oidn::Format::Float3, resolution.x(), resolution.y());
        filter.setImage("normal", normals_buf, oidn::Format::Float3, resolution.x(), resolution.y());
        filter.setImage("albedo", albedo_buf, oidn::Format::Float3, resolution.x(), resolution.y());
        filter.setImage("output", m_output->data(), oidn::Format::Float3, resolution.x(), resolution.y());
        filter.set("hdr", true);
        filter.commit();

        filter.execute();
        
        m_output->save();

        const char* errorMessage;
        if (device.getError(errorMessage) != oidn::Error::None) std::cout << "Error: " << errorMessage << std::endl;
    }
 
    std::string toString() const override {
        return tfm::format(
            "Denoise"
        );
    }
};

}

REGISTER_POSTPROCESS(Denoise, "denoising");

#endif
