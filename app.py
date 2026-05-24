import streamlit as st
from PIL import Image
import torch
import io
import zipfile
from pathlib import Path
from streamlit_image_comparison import image_comparison
from PIL import ImageFilter
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BEN2 · Background Remover",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from ben2 import BEN_Base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BEN_Base.from_pretrained("PramaLLC/BEN2")
    model.to(device).eval()
    return model, device


def remove_background(model, device, image: Image.Image) -> Image.Image:
    """Run BEN2 inference and return RGBA image."""
    result = model.inference(image)
    if isinstance(result, (list, tuple)):
        result = result[0]
    return result


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 >BEN2<br>Background Remover</h1>', unsafe_allow_html=True)
st.markdown('<h2>Neural background removal · PramaLLC/BEN2</h2>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# st.markdown('<p class="main-title">BEN2<br>Background Remover</p>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle">Neural background removal · PramaLLC/BEN2</p>', unsafe_allow_html=True)
# st.markdown("<hr>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    output_format = st.selectbox(
        "Output format",
        ["PNG (transparent)", "WebP (transparent)"],
        index=0,
    )
    fmt_map = {"PNG (transparent)": "PNG", "WebP (transparent)": "WEBP"}
    out_fmt = fmt_map[output_format]
    out_ext = out_fmt.lower().replace("webp", "webp").replace("png", "png")

    st.markdown("---")
    st.markdown("### ℹ️ Info")
    device_label = "CUDA 🚀" if torch.cuda.is_available() else "CPU 🐢"
    st.markdown(f"**Device:** `{device_label}`")
    st.markdown("**Model:** `PramaLLC/BEN2`")

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.7rem;color:#4b5563;">Upload one or more images, '
        "click <b>Remove Backgrounds</b>, then download individual results "
        "or grab the ZIP archive.</p>",
        unsafe_allow_html=True,
    )

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop images here or click to browse",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded:
    st.info("⬆️  Upload one or more images to get started.")
    st.stop()

st.markdown(f"**{len(uploaded)} image(s) selected**")

# ── Process button ────────────────────────────────────────────────────────────
run = st.button("✂️  Remove Backgrounds", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = {}

if run:
    with st.spinner("Loading model…"):
        model, device = load_model()

    progress = st.progress(0, text="Processing…")
    results = {}

    for i, f in enumerate(uploaded):
        progress.progress((i) / len(uploaded), text=f"Processing {f.name}…")
        img = Image.open(f).convert("RGBA")
        out_img = remove_background(model, device, img)
        results[f.name] = out_img
        progress.progress((i + 1) / len(uploaded), text=f"Done: {f.name}")

    progress.empty()
    st.session_state.results = results
    st.success(f"✅ Processed {len(results)} image(s)!")

# ── Results grid ──────────────────────────────────────────────────────────────
if st.session_state.results:
    st.markdown("---")
    st.markdown("### Results")

    cols_per_row = 2
    items = list(zip(uploaded, [st.session_state.results.get(f.name) for f in uploaded]))
    items = [(f, r) for f, r in items if r is not None]

    for row_start in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row, gap="large")
        # for col_idx, (f, result_img) in enumerate(items[row_start: row_start + cols_per_row]):
        #     with cols[col_idx]:
        for col_idx, (f, result_img) in enumerate(items[row_start: row_start + cols_per_row]):
            item_idx = row_start + col_idx
            with cols[col_idx]:
                orig = Image.open(f)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<p class="image-label">Original</p>', unsafe_allow_html=True)
                    st.image(orig, use_container_width=True)
                with c2:
                    st.markdown('<p class="image-label">Result</p>', unsafe_allow_html=True)
                    # Show on checkered-like dark bg
                    st.image(result_img, use_container_width=True)

                # stem = Path(f.name).stem
                # out_name = f"{stem}_no_bg.{out_ext}"
                # st.download_button(
                #     label=f"⬇ Download {out_name}",
                #     data=pil_to_bytes(result_img, out_fmt),
                #     file_name=out_name,
                #     mime=f"image/{out_ext}",
                #     # key=f"dl_{f.name}",
                #     key=f"dl_{item_idx}_{f.name}",
                #     use_container_width=True,
                # )

                # Comparison slider
                st.markdown('<p class="image-label">Before / After</p>', unsafe_allow_html=True)
                orig_rgb = orig.convert("RGB")

                # Накладаємо результат на білий фон щоб прозорість була видна
                white_bg = Image.new("RGB", result_img.size, (255, 255, 255))
                white_bg.paste(result_img, mask=result_img.split()[3])

                # result_rgb = result_img.convert("RGB")
                image_comparison(
                    img1=orig_rgb,
                    img2=white_bg,
                    label1="Original",
                    label2="No BG",
                    width=350,
                    # key=f"cmp_{f.name}",
                )

                alpha = np.array(result_img.split()[3])
                coverage = (alpha > 128).sum() / alpha.size * 100
                # edges = np.array(Image.fromarray(alpha).filter(ImageFilter.FIND_EDGES))
                # sharpness = edges.mean()

                edges = np.array(Image.fromarray(alpha).filter(ImageFilter.FIND_EDGES))

                # Кількість крайових пікселів (де є перехід)
                edge_pixels = (edges > 10).sum()

                # Нормалізуємо на периметр, а не площу
                sharpness = edges[edges > 10].mean() if edge_pixels > 0 else 0
    
                m1, m2 = st.columns(2)
                m1.metric("Об'єкт", f"{coverage:.1f}%")
                m2.metric("Чіткість країв", f"{sharpness:.1f}")


                stem = Path(f.name).stem
                out_name = f"{stem}_no_bg.{out_ext}"
                st.download_button(
                    label=f"⬇ Download {out_name}",
                    data=pil_to_bytes(result_img, out_fmt),
                    file_name=out_name,
                    mime=f"image/{out_ext}",
                    key=f"dl_{f.name}",
                    use_container_width=True,
                )

    # ── ZIP download ──────────────────────────────────────────────────────────
    if len(items) > 1:
        st.markdown("---")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f, result_img in items:
                stem = Path(f.name).stem
                out_name = f"{stem}_no_bg.{out_ext}"
                zf.writestr(out_name, pil_to_bytes(result_img, out_fmt))
        st.download_button(
            label="📦  Download All as ZIP",
            data=zip_buf.getvalue(),
            file_name="removed_backgrounds.zip",
            mime="application/zip",
            use_container_width=True,
        )
