import json
import sys

from PIL import Image, ImageDraw, ImageFont


def hex_color(value, alpha=255):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def scaled_points(points, scale):
    return [(scale * x, scale * y) for x, y in points]


def main():
    with open(sys.argv[1], "r") as handle:
        spec = json.load(handle)

    scale = spec.get("scale", 3)
    width = spec["width"]
    height = spec["height"]
    image = Image.new("RGBA", (scale * width, scale * height), hex_color("#ffffff"))
    draw = ImageDraw.Draw(image)

    for item in spec["items"]:
        kind = item["type"]

        if kind == "polygon":
            fill = hex_color(item["fill"], round(255 * item.get("fill_alpha", 1)))
            outline = hex_color(item.get("outline", "#000000"))
            points = scaled_points(item["points"], scale)
            draw.polygon(points, fill=fill, outline=outline)

        elif kind == "circle":
            x, y = item["center"]
            r = item["radius"]
            box = [scale * (x - r), scale * (y - r), scale * (x + r), scale * (y + r)]
            draw.ellipse(
                box,
                fill=hex_color(item["fill"]),
                outline=hex_color(item.get("outline", item["fill"])),
                width=max(1, round(scale * item.get("width", 1))),
            )

        elif kind == "text":
            font = ImageFont.load_default(size=scale * item.get("size", 14))
            x, y = item["position"]
            draw.text((scale * x, scale * y), item["text"], fill=hex_color(item.get("fill", "#000000")), font=font)

    image = image.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
    image.save(spec["output"])


if __name__ == "__main__":
    main()
