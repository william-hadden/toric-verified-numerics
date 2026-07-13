import json
import math
import sys

from PIL import Image, ImageDraw, ImageFont


def hex_color(value, alpha=255):
    """Convert a CSS-style hexadecimal color to an RGBA tuple."""
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def scaled_points(points, scale):
    """Scale logical drawing points to the antialiased canvas."""
    return [(scale * x, scale * y) for x, y in points]


def scaled_width(item, scale):
    """Return a positive antialiased stroke width for a drawing item."""
    return max(1, round(scale * item.get("width", 1)))


def field_color(value):
    """Map a signed normalized value to the plot's diverging RGBA color."""
    blue, white, red = (33, 102, 172), (247, 247, 247), (178, 24, 43)
    value = min(1.0, max(-1.0, value))
    left, right, fraction = (
        (blue, white, value + 1.0) if value < 0 else (white, red, value)
    )
    rgb = tuple(
        round((1.0 - fraction) * a + fraction * b)
        for a, b in zip(left, right)
    )
    return rgb + (255,)


def maximum_side_length(points):
    """Return the longest side length of a screen-space triangle."""
    sides = (
        math.dist(points[0], points[1]),
        math.dist(points[1], points[2]),
        math.dist(points[2], points[0]),
    )
    return max(sides)


def midpoint(left, right):
    """Return the coordinate-wise midpoint of two tuples."""
    return tuple((a + b) / 2 for a, b in zip(left, right))


def draw_affine_triangle(draw, points, values, limit, maximum_diameter):
    """Rasterize one affine element by recursive midpoint subdivision."""
    if maximum_side_length(points) <= maximum_diameter:
        color = field_color(sum(values) / (3.0 * limit))
        draw.polygon(points, fill=color)
        return
    p12 = midpoint(points[0], points[1])
    p23 = midpoint(points[1], points[2])
    p31 = midpoint(points[2], points[0])
    v12 = (values[0] + values[1]) / 2
    v23 = (values[1] + values[2]) / 2
    v31 = (values[2] + values[0]) / 2
    children = (
        ((points[0], p12, p31), (values[0], v12, v31)),
        ((p12, points[1], p23), (v12, values[1], v23)),
        ((p31, p23, points[2]), (v31, v23, values[2])),
        ((p12, p23, p31), (v12, v23, v31)),
    )
    for child_points, child_values in children:
        draw_affine_triangle(draw, child_points, child_values, limit, maximum_diameter)


def draw_field(draw, item, scale):
    """Draw all independently affine elements of a CR finite element field."""
    limit = item["limit"]
    maximum_diameter = scale * item["maximum_leaf_diameter"]
    for points, values in zip(item["triangles"], item["values"]):
        draw_affine_triangle(
            draw,
            scaled_points(points, scale),
            values,
            limit,
            maximum_diameter,
        )


def main():
    """Render the drawing specification supplied by the Julia driver."""
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
            outline = hex_color(item["outline"]) if "outline" in item else None
            points = scaled_points(item["points"], scale)
            draw.polygon(points, fill=fill, outline=outline, width=scaled_width(item, scale))

        elif kind == "line":
            draw.line(
                scaled_points(item["points"], scale),
                fill=hex_color(item.get("fill", "#000000")),
                width=scaled_width(item, scale),
                joint="curve",
            )

        elif kind == "field":
            draw_field(draw, item, scale)

        elif kind == "circle":
            x, y = item["center"]
            r = item["radius"]
            box = [scale * (x - r), scale * (y - r), scale * (x + r), scale * (y + r)]
            draw.ellipse(
                box,
                fill=hex_color(item["fill"]),
                outline=hex_color(item.get("outline", item["fill"])),
                width=scaled_width(item, scale),
            )

        elif kind == "text":
            font = ImageFont.load_default(size=scale * item.get("size", 14))
            x, y = item["position"]
            draw.text(
                (scale * x, scale * y),
                item["text"],
                fill=hex_color(item.get("fill", "#000000")),
                font=font,
            )

    image = image.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
    image.save(spec["output"])


if __name__ == "__main__":
    main()
