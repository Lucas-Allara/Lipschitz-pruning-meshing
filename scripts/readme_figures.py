"""Writes the images of README.md into figures/readme, from the poster and its figures.

    python scripts/readme_figures.py

hero.jpg is the band of the poster with the three scenes and the colorbar, poster.jpg the whole
poster, scaling.png its plot of evaluations against the number of spheres (pypdfium2 renders the
PDF pages: pip install pypdfium2), coarse_uniform.jpg and coarse_coarse.jpg the detail of part 145
in poster/figures, dynamic.jpg its frame of the patch on Monument, and social-preview.jpg the
1280 x 640 card that GitHub shows where the repository is linked (Settings, Social preview), cut
from the header of the poster.

The two clips are screen recordings of the viewer, cropped to the scene and encoded as animated
WebP at about 20 frames per second:

    viewer.webp   ./build/lpm_view monument 8 --colors --vsync --bench 6000
    coarse.webp   ./build/lpm_view scenes/inversecsg/145.json 8 --coarse --edges --vsync --bench 3000
                  (frames 0 to 64 played forward and back, while the sphere is in view)

    ffmpeg -framerate 21 -i f_%03d.ppm -vf "crop=W:H:X:Y,scale=760:-2:flags=lanczos" \\
           -c:v libwebp_anim -q:v 72 -compression_level 6 -loop 0 viewer.webp
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
POSTER = ROOT / "poster"
OUT = ROOT / "figures" / "readme"
# The poster is 1984.25 x 2834.65 points; the header ends at 372 points and the teaser band at 955.
HEADER_END, TEASER_END = 372, 955


def flat(im: Image.Image) -> Image.Image:
    """The image on white, without alpha."""
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def save(im: Image.Image, name: str, width=None, quality: int = 88):
    im = flat(im)
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    path = OUT / name
    if path.suffix == ".png":
        im.save(path, optimize=True)
    else:
        im.save(path, quality=quality, optimize=True, progressive=True)
    print(f"{path.relative_to(ROOT)}: {im.width} x {im.height}, {path.stat().st_size // 1024} KB")


def pdf_page(path: Path, width: int) -> Image.Image:
    import pypdfium2 as pdfium
    page = pdfium.PdfDocument(str(path))[0]
    return page.render(scale=width / page.get_size()[0]).to_pil()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    poster = pdf_page(POSTER / "poster.pdf", 2400)
    s = poster.width / 1984.25
    save(poster, "poster.jpg", 1200, 90)
    save(poster.crop((0, round(HEADER_END * s), poster.width, round(TEASER_END * s))), "hero.jpg", 1800)
    # The social card: the header and the teaser at 1280 px, with the burgundy band up to the top.
    card = flat(poster.crop((0, 0, poster.width, round(TEASER_END * s))))
    card = card.resize((1280, round(card.height * 1280 / card.width)), Image.LANCZOS)
    canvas = Image.new("RGB", (1280, 640), card.getpixel((4, 4)))
    canvas.paste(card, (0, 640 - card.height))
    save(canvas, "social-preview.jpg", None, 90)
    save(pdf_page(POSTER / "figures" / "scaling.pdf", 1200), "scaling.png")
    save(Image.open(POSTER / "figures" / "dynamic.png"), "dynamic.jpg", 1400)
    save(Image.open(POSTER / "figures" / "coarse_uniform_zoom.png"), "coarse_uniform.jpg", 1000)
    save(Image.open(POSTER / "figures" / "coarse_coarse_zoom.png"), "coarse_coarse.jpg", 1000)


if __name__ == "__main__":
    main()
