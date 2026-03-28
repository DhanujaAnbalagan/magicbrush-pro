# 🎨 MagicBrush Pro: Self-Guided Contextual Photo Editing

## 🚀 Overview

MagicBrush Pro is a **Generative AI-powered image editing system** that performs **localized, instruction-based edits** using natural language prompts.

Unlike traditional tools, this system:

* Edits only the **user-selected region**
* Keeps the **rest of the image pixel-perfect unchanged**
* Automatically selects the **best output from multiple generations**

The core idea is simple but powerful:
👉 *Controlled generative editing with strict background preservation.*

---

## 🔥 Key Features

* 🧠 **Natural Language Editing**
  Modify images using prompts like:
  `"change the red shirt to a formal blue silk shirt"`

* 🎯 **Mask-Based Local Editing**
  User-defined mask ensures edits are restricted to specific regions

* 🖼️ **Pixel-Perfect Background Preservation**
  Hard compositing guarantees zero background distortion

* ⚡ **Multi-Sample Generation (N = 4)**
  Generates multiple outputs and selects the best automatically

* 📊 **Hybrid Scoring System**
  Combines semantic relevance and perceptual realism

* 🧩 **Seamless Blending**
  No visible edges or cut-paste artifacts

---

## 🧠 System Architecture

### Core Components

1. **Stable Diffusion Inpainting**

   * Generates new content inside masked regions

2. **CLIP (ViT-B/32)**

   * Aligns generated output with text prompt

3. **VAE (Latent Space Processing)**

   * Compresses image to 64×64 latent space for efficiency

4. **UNet (Diffusion Model)**

   * Performs iterative denoising for image generation

---

## ⚙️ Pipeline Workflow

1. **Input**

   * Upload image
   * Draw mask (editable region)

2. **Preprocessing**

   * Resize to 512×512
   * Mask refinement (dilation + thresholding)

3. **Generation**

   * Batch generation (N = 4 samples)

4. **Evaluation**

   * CLIP → semantic relevance
   * LPIPS → perceptual similarity

5. **Scoring**

   ```
   Final Score = 0.7 × CLIP − 0.3 × LPIPS
   ```

6. **Compositing**

   ```
   Result = Generated × Mask + Original × (1 − Mask)
   ```

7. **Output**

   * Best result selected automatically
   * All candidates shown in gallery

---

## 📊 Performance Metrics

| Metric         | Value  | Interpretation                        |
| -------------- | ------ | ------------------------------------- |
| Precision@4    | 0.95   | At least one good result in 95% cases |
| Avg CLIP Score | 28.77  | Strong prompt alignment               |
| Avg LPIPS      | 0.3202 | High perceptual similarity            |
| Inference Time | ~3.2s  | 4-image batch on T4 GPU               |

---

## 🧪 Key Insights

* **Multi-sample generation is essential**
  Single output (N=1) is unreliable due to stochastic nature

* **CLIP alone is not enough**
  It can produce unrealistic edits → LPIPS fixes this

* **Hard compositing is critical**
  Without it, background corruption is unavoidable

* **Latency vs Quality tradeoff is optimized**
  Achieves near real-time performance with high-quality output

---

## 🛠️ Tech Stack

### Backend

* Python
* PyTorch
* FastAPI
* HuggingFace Diffusers

### Models

* Stable Diffusion (Inpainting)
* CLIP (ViT-B/32)
* LPIPS (VGG)

### Frontend

* HTML, CSS, JavaScript
* Canvas API (Mask Drawing)

### Image Processing

* OpenCV
* Pillow
* NumPy

### Deployment

* Docker
* Ngrok / Cloud / Local GPU

---

## 💻 Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd MagicBrush-Pro
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Backend

```bash
uvicorn main:app --reload
```

### 4. Open Frontend

Open `index.html` in browser or connect via API.

---

## ⚠️ Limitations

* 🔻 Resolution capped at **512×512**
* 💻 Requires **GPU (8GB+ VRAM recommended)**
* 📏 Weak geometric control (size/shape inconsistencies possible)

---

## 🔮 Future Improvements

* ControlNet → better structural control
* SAM → auto mask generation
* Real-ESRGAN → high-resolution upscaling
* SDXL → improved realism
* Multi-style model switching

---

## 📌 Conclusion

MagicBrush Pro is not just another image editor.
It’s a **controlled generative system** that balances:

* Creativity (Diffusion)
* Accuracy (CLIP)
* Realism (LPIPS)
* Reliability (Hard Compositing)
