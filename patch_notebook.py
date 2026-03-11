import json

with open('backend_colab.ipynb', 'r') as f:
    nb = json.load(f)

# Updated Cell 5: main.py — hard pixel-level compositing after SD Inpainting
new_main_source = '''main_code = """from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from utils import decode_base64_image, encode_image_base64, preprocess_image
from scoring import get_scorer
from model_pipeline import get_pipeline
import traceback
import base64
import io
import numpy as np
from PIL import Image, ImageFilter

app = FastAPI(title="MagicBrush Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def decode_mask(mask_b64: str, target_size=(512, 512)) -> np.ndarray:
    """Decode base64 mask canvas → float32 numpy array [0.0, 1.0].
    The canvas stores painted strokes in the alpha channel.
    Returns shape (H, W, 1) float32 — 1.0 = edit here, 0.0 = keep original.
    """
    if "," in mask_b64:
        mask_b64 = mask_b64.split(",")[1]
    data    = base64.b64decode(mask_b64)
    img     = Image.open(io.BytesIO(data)).convert("RGBA")
    img     = img.resize(target_size, Image.Resampling.LANCZOS)
    alpha   = np.array(img)[:, :, 3].astype(np.float32) / 255.0
    # Soft threshold: anything painted (alpha > 0.1) counts as mask
    alpha   = (alpha > 0.1).astype(np.float32)
    # Slight dilation to cover brush edge antialiasing
    pil_a   = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    pil_a   = pil_a.filter(ImageFilter.MaxFilter(5))
    alpha   = np.array(pil_a).astype(np.float32) / 255.0
    return alpha[:, :, np.newaxis]   # (H, W, 1)

def hard_composite(original, generated, mask_alpha):
    """
    Pixel-perfect compositing:  result = generated * mask + original * (1 - mask)
    original, generated : PIL RGB images at 512x512
    mask_alpha          : numpy float32 (H, W, 1)  values 0..1
    """
    orig_arr = np.array(original.resize((512,512)).convert("RGB")).astype(np.float32)
    gen_arr  = np.array(generated.resize((512,512)).convert("RGB")).astype(np.float32)
    blended  = gen_arr * mask_alpha + orig_arr * (1.0 - mask_alpha)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))

def make_full_mask_alpha(size=(512, 512)):
    return np.ones((*size, 1), dtype=np.float32)

@app.on_event("startup")
async def startup_event():
    print("Initializing models...")
    get_pipeline()
    get_scorer()
    print("Application startup complete.")

@app.post("/generate")
async def generate_images(
    image:  str = Form(...),
    prompt: str = Form(...),
    mask:   str = Form(None),
):
    try:
        orig_img = decode_base64_image(image)
        proc_img = preprocess_image(orig_img)   # 512x512 RGB PIL

        # --- Decode mask ---
        if mask and len(mask) > 100:
            print("Mask received — decoding painted region...")
            mask_alpha = decode_mask(mask, target_size=(512, 512))
            # Convert to PIL L-mode for SD pipeline (white=inpaint, black=keep)
            mask_pil   = Image.fromarray((mask_alpha[:,:,0] * 255).astype(np.uint8), mode="L")
        else:
            print("No mask — inpainting full image.")
            mask_alpha = make_full_mask_alpha()
            mask_pil   = Image.new("L", (512, 512), 255)

        # --- Run SD Inpainting ---
        pipeline = get_pipeline()
        generated_images = pipeline.generate(proc_img, mask_pil, prompt, num_samples=4)

        # --- HARD COMPOSITE: enforce exact painted boundary ---
        # SD Inpainting bleeds slightly outside the mask. This step ensures
        # ONLY pixels inside the painted brush strokes are replaced.
        composited_images = [hard_composite(proc_img, g, mask_alpha) for g in generated_images]

        # --- Score & rank ---
        scorer  = get_scorer()
        results = []
        for g_img in composited_images:
            scores  = scorer.score(proc_img, g_img, prompt)
            img_b64 = encode_image_base64(g_img)
            results.append({"image": img_b64, "scores": scores})

        results.sort(key=lambda x: x["scores"]["final_score"], reverse=True)
        return {"samples": results}

    except Exception as e:
        print("Error during generation:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
"""

with open('main.py', 'w') as f:
    f.write(main_code)

print('main.py created with hard pixel-level mask compositing.')
'''

nb['cells'][5]['source'] = [new_main_source]

with open('backend_colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Notebook Cell 5 updated with strict mask enforcement.')
