"""Procedural textures for the construction-site scene (written once as PNG files)."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

TEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "textures")


def _value_noise(n, scale, rng):
    """Smooth value noise on an n x n grid with feature size `scale` pixels (tileable)."""
    g = max(2, n // scale)
    base = rng.random((g, g))
    # tileable bicubic-like upsampling via FFT-free separable interpolation
    x = np.linspace(0, g, n, endpoint=False)
    i0 = np.floor(x).astype(int) % g
    i1 = (i0 + 1) % g
    t = x - np.floor(x)
    t = t * t * (3 - 2 * t)
    a = base[i0][:, i0] * (1 - t)[None, :] + base[i0][:, i1] * t[None, :]
    b = base[i1][:, i0] * (1 - t)[None, :] + base[i1][:, i1] * t[None, :]
    return a * (1 - t)[:, None] + b * t[:, None]


def fractal(n, rng, octaves=((128, 0.5), (48, 0.25), (16, 0.15), (5, 0.1))):
    out = np.zeros((n, n))
    for s, w in octaves:
        out += w * _value_noise(n, s, rng)
    out -= out.min()
    return out / out.max()


def ground_texture(path, n=1024, seed=1):
    """Compacted construction-site soil with gravel and faint tyre tracks."""
    rng = np.random.default_rng(seed)
    f = fractal(n, rng)
    base = np.array([118, 104, 86], float)          # sandy brown-grey soil
    dark = np.array([78, 70, 60], float)
    img = base[None, None, :] * (0.75 + 0.35 * f[..., None]) * 0.95
    patches = fractal(n, rng, ((256, 0.7), (96, 0.3)))
    img = img * (1 - 0.25 * (patches[..., None] > 0.62)) + dark * 0.25 * (patches[..., None] > 0.62)
    # gravel speckles
    ns = n * n // 18
    ys, xs = rng.integers(0, n, ns), rng.integers(0, n, ns)
    tone = rng.uniform(0.55, 1.35, ns)
    col = np.array([132, 126, 118], float)
    for dy in (0, 1):
        for dx in (0, 1):
            img[(ys + dy) % n, (xs + dx) % n] = np.clip(col[None, :] * tone[:, None], 0, 255)
    # faint tyre tracks along x
    yy = np.arange(n)
    for c0 in (0.31, 0.37, 0.71, 0.77):
        band = np.exp(-((yy / n - c0) ** 2) / (2 * 0.008 ** 2))
        img *= (1 - 0.10 * band[:, None, None])
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    Image.fromarray(img).filter(ImageFilter.SMOOTH).save(path)


def concrete_texture(path, n=512, seed=2):
    rng = np.random.default_rng(seed)
    f = fractal(n, rng, ((96, 0.5), (24, 0.3), (6, 0.2)))
    img = np.array([168, 166, 160], float)[None, None, :] * (0.82 + 0.25 * f[..., None])
    ns = n * n // 40
    ys, xs = rng.integers(0, n, ns), rng.integers(0, n, ns)
    img[ys, xs] *= rng.uniform(0.6, 0.9, (ns, 1))
    img[:, :3] *= 0.75
    img[:3, :] *= 0.75
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def wood_texture(path, n=512, seed=3):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)[None, :]
    y = np.linspace(0, 1, n)[:, None]
    warp = fractal(n, rng, ((128, 0.6), (32, 0.4)))
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (y * 26 + 2.5 * warp))
    base = np.array([150, 112, 72], float)
    img = base[None, None, :] * (0.78 + 0.28 * rings[..., None]) * (0.9 + 0.15 * fractal(n, rng)[..., None])
    img = np.broadcast_to(img, (n, n, 3)).copy()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def steel_primer_texture(path, n=1024, seed=4, label=None):
    """Red-oxide primer steel with slight mottling, weld seams and an optional stencil label."""
    rng = np.random.default_rng(seed)
    f = fractal(n, rng, ((160, 0.5), (40, 0.3), (8, 0.2)))
    base = np.array([158, 58, 36], float)
    img = base[None, None, :] * (0.82 + 0.25 * f[..., None])
    img[:, :6] *= 0.6
    img[:, -6:] *= 0.6
    img[:6, :] *= 0.6
    img[-6:, :] *= 0.6
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if label:
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(n * 0.10))
        except OSError:
            font = ImageFont.load_default()
        d.text((int(n * 0.06), int(n * 0.38)), label, fill=(236, 232, 220), font=font)
        # the stencil boxes show the texture mirrored on their outward faces: mirror it so that it reads correctly
        im = ImageOps.mirror(im)
    im.save(path)


def aluminium_texture(path, n=1024, seed=6, label=None):
    """Brushed aluminium (mill finish) with faint streaks along the beam and an optional printed label."""
    rng = np.random.default_rng(seed)
    streak = _value_noise(n, 3, rng)[0]                         # fine streaks across the rows
    rows = 0.6 * streak[:, None] + 0.4 * _value_noise(n, 2, rng)
    mott = fractal(n, rng, ((256, 0.6), (64, 0.4)))
    base = np.array([158, 165, 174], float)
    img = base[None, None, :] * (0.90 + 0.06 * rows[..., None] + 0.06 * mott[..., None])
    img[:, :6] *= 0.8
    img[:, -6:] *= 0.8
    img[:6, :] *= 0.8
    img[-6:, :] *= 0.8
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if label:
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(n * 0.10))
        except OSError:
            font = ImageFont.load_default()
        d.text((int(n * 0.06), int(n * 0.38)), label, fill=(52, 56, 64), font=font)
        # the stencil boxes show the texture mirrored on their outward faces: mirror it so that it reads correctly
        im = ImageOps.mirror(im)
    im.save(path)


def corrugated_texture(path, n=512, rgb=(52, 92, 140), seed=5):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)[None, :]
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * 24 * x)
    img = np.array(rgb, float)[None, None, :] * (0.72 + 0.36 * stripes[..., None])
    img = img * (0.92 + 0.12 * fractal(n, rng)[..., None])
    img = np.broadcast_to(img, (n, n, 3)).copy()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def hazard_texture(path, n=256):
    img = np.zeros((n, n, 3), np.uint8)
    yy, xx = np.mgrid[0:n, 0:n]
    band = ((xx + yy) // (n // 6)) % 2 == 0
    img[band] = (240, 196, 20)
    img[~band] = (30, 30, 30)
    Image.fromarray(img).save(path)


def cone_texture(path, n=256):
    img = np.zeros((n, n, 3), np.uint8)
    img[:] = (236, 90, 20)
    rows = np.arange(n)
    for a, b in ((0.30, 0.42), (0.55, 0.64)):
        img[(rows >= a * n) & (rows < b * n)] = (235, 235, 230)
    Image.fromarray(img).save(path)


def sky_texture(path, w=512, h=3072):
    """Vertical gradient skybox strip (MuJoCo 'skybox' expects a 1x6 strip layout)."""
    img = np.zeros((h, w, 3), float)
    top = np.array([96, 146, 205], float)
    hor = np.array([214, 226, 236], float)
    face = h // 6
    for k in range(6):
        y = np.linspace(0, 1, face)[:, None]
        if k == 2:      # up
            img[k * face:(k + 1) * face] = top
        elif k == 3:    # down
            img[k * face:(k + 1) * face] = np.array([150, 140, 125], float)
        else:
            t = np.clip(y * 1.25, 0, 1) ** 0.8
            img[k * face:(k + 1) * face] = (top * (1 - t) + hor * t)[:, None, :]
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def label_name(label, material="steel"):
    """File name of the stencil-label texture of a payload (one file per label and material)."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")
    return f"{material}_label_{slug}.png"


def _atomic(fn, path):
    """Write a texture to a temporary file and move it into place (safe with parallel runs)."""
    tmp = f"{path}.{os.getpid()}.tmp.png"
    fn(tmp)
    os.replace(tmp, path)


def make_all(label="AL 280   63 kg", material="aluminium"):
    os.makedirs(TEX_DIR, exist_ok=True)
    jobs = {
        "ground.png": lambda p: ground_texture(p),
        "concrete.png": lambda p: concrete_texture(p),
        "wood.png": lambda p: wood_texture(p),
        "steel_primer.png": lambda p: steel_primer_texture(p),
        "aluminium.png": lambda p: aluminium_texture(p),
        "container.png": lambda p: corrugated_texture(p),
        "hazard.png": lambda p: hazard_texture(p),
        "cone.png": lambda p: cone_texture(p),
    }
    for name, fn in jobs.items():
        p = os.path.join(TEX_DIR, name)
        if not os.path.exists(p):
            _atomic(fn, p)
    # the stencil label depends on the payload
    p = os.path.join(TEX_DIR, label_name(label, material))
    if not os.path.exists(p):
        fn = aluminium_texture if material == "aluminium" else steel_primer_texture
        _atomic(lambda q: fn(q, label=label), p)
    return TEX_DIR


if __name__ == "__main__":
    print(make_all())
