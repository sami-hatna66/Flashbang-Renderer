#include <lightwave.hpp>

namespace lightwave {

class ImageTexture : public Texture {
    enum class BorderMode {
        Clamp,
        Repeat,
    };

    enum class FilterMode {
        Nearest,
        Bilinear,
    };

    ref<Image> m_image;
    float m_exposure;
    BorderMode m_border;
    FilterMode m_filter;

public:
    ImageTexture(const Properties &properties) {
        if (properties.has("filename")) {
            m_image = std::make_shared<Image>(properties);
        } else {
            m_image = properties.getChild<Image>();
        }
        m_exposure = properties.get<float>("exposure", 1);

        m_border =
            properties.getEnum<BorderMode>("border", BorderMode::Repeat,
                                           {
                                               { "clamp", BorderMode::Clamp },
                                               { "repeat", BorderMode::Repeat },
                                           });

        m_filter = properties.getEnum<FilterMode>(
            "filter", FilterMode::Bilinear,
            {
                { "nearest", FilterMode::Nearest },
                { "bilinear", FilterMode::Bilinear },
            });
    }

    Color evaluate(const Point2 &uv) const override {
        Point2 adjusted_uv;
        if (m_border == BorderMode::Clamp) {
            adjusted_uv = {clamp(uv.x(), 0.f, 1.f), clamp(1.f - uv.y(), 0.f, 1.f)};
        } else {
            adjusted_uv = {fmod(uv.x(), 1.f), fmod(1.f - uv.y(), 1.f)};
            if (adjusted_uv.x() < 0.f) adjusted_uv.x() += 1.f;
            if (adjusted_uv.y() < 0.f) adjusted_uv.y() += 1.f;
        }

        auto width = m_image->resolution().x();
        auto height = m_image->resolution().y();

        if (m_filter == FilterMode::Nearest) {
            int i = min(floor(adjusted_uv.x() * width), width - 1);
            int j = min(floor(adjusted_uv.y() * height), height - 1);
            return m_image->get({i, j}) * m_exposure;
        } else {
            float scaledX = (adjusted_uv.x() * width) - 0.5f;
            float scaledY = (adjusted_uv.y() * height) - 0.5f;

            int x0 = floor(scaledX);
            int x1 = x0 + 1;
            int y0 = floor(scaledY);
            int y1 = y0 + 1;

            x0 = clamp(x0, 0, width - 1);
            x1 = clamp(x1, 0, width - 1);
            y0 = clamp(y0, 0, height - 1);
            y1 = clamp(y1, 0, height - 1);

            float weightX1 = scaledX - x0;
            float weightX0 = 1.f - weightX1;
            float weightY1 = scaledY - y0;
            float weightY0 = 1.f - weightY1;

            Color interpolatedValue = weightX0 * weightY0 * m_image->get({x0, y0}) + 
                                      weightX0 * weightY1 * m_image->get({x0, y1}) + 
                                      weightX1 * weightY0 * m_image->get({x1, y0}) + 
                                      weightX1 * weightY1 * m_image->get({x1, y1});

            return interpolatedValue * m_exposure;
        }
    }

    std::string toString() const override {
        return tfm::format("ImageTexture[\n"
                           "  image = %s,\n"
                           "  exposure = %f,\n"
                           "]",
                           indent(m_image), m_exposure);
    }
};

} // namespace lightwave

REGISTER_TEXTURE(ImageTexture, "image")
