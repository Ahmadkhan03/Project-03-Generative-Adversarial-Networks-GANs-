# GenAI Assignment 03 — GANs for Image Synthesis

> **Course:** Generative AI (AI4009) | Spring 2026
> **University:** FAST National University of Computer and Emerging Sciences
> **Team:** Rana M. Ahmad (22F-8758) & Urwa Sajid (22F-3244)

---

## What This Project Does

Three GAN-based image synthesis tasks, each with an interactive Streamlit demo and a unified combined dashboard for quick presentation and grading.

| # | Model | Task |
|---|-------|------|
| Q1 | DCGAN vs WGAN-GP | Generate anime-style faces; compare training stability and mode collapse |
| Q2 | Pix2Pix (cGAN) | Translate sketch images → realistic/colorized outputs |
| Q3 | CycleGAN | Unpaired domain translation between Sketch ↔ Photo style |

---

## Project Structure

```
GenAI-Assignment03/
├── App.py                          # Combined dashboard (all 3 questions)
├── app_q1.py                       # DCGAN vs WGAN-GP standalone
├── app_q2.py                       # Pix2Pix standalone
├── app_q3.py                       # CycleGAN standalone
├── requirements.txt
├── .streamlit/
│   └── config.toml                 # Upload size config for large checkpoints
├── output/
│   ├── dcgan_generator_final.pth
│   ├── wgan_generator_final.pth
│   ├── pix2pix_generator_final.pth
│   └── pix2pix_discriminator_final.pth
└── cyclegan_weights.pt
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Bilxl99/GenAI-Assignment03.git
cd GenAI-Assignment03

# 2. Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Apps

### Recommended — Combined Dashboard

```bash
streamlit run App.py
```

The sidebar lets you switch between:
- **Home** — project overview and model status
- **Q1** — side-by-side DCGAN and WGAN-GP image grids
- **Q2** — single or batch sketch translation with download support
- **Q3** — bidirectional CycleGAN translation with cycle reconstruction and SSIM/PSNR metrics

### Standalone Demos

```bash
streamlit run app_q1.py
streamlit run app_q2.py
streamlit run app_q3.py
```

---

## Model Checkpoints

Default paths expected by the apps:

| Model | Path |
|-------|------|
| DCGAN Generator | `output/dcgan_generator_final.pth` |
| WGAN-GP Generator | `output/wgan_generator_final.pth` |
| Pix2Pix Generator | `output/pix2pix_generator_final.pth` |
| CycleGAN | `cyclegan_weights.pt` |

You can override paths from the sidebar or upload checkpoint files directly through the UI. Large `.pth` / `.pt` files are supported — the Streamlit config allows up to 4 GB uploads.

---

## Key Concepts

**Q1 — DCGAN vs WGAN-GP**
DCGAN uses standard GAN loss (binary cross-entropy) which is prone to mode collapse and training instability. WGAN-GP replaces this with Wasserstein distance + gradient penalty, producing more diverse and stable outputs.

**Q2 — Pix2Pix**
A conditional GAN where both generator and discriminator receive the input sketch as context. Uses a U-Net generator for fine-grained reconstruction and a PatchGAN discriminator. L1 loss is added alongside adversarial loss to preserve structural detail.

**Q3 — CycleGAN**
Enables unpaired image-to-image translation using two generators and two discriminators. Cycle consistency loss ensures `A → B → A` and `B → A → B` reconstructions are faithful, removing the need for paired training data.

---

## Dependencies

```
streamlit
torch
torchvision
numpy
Pillow
scikit-image
```

Full version list in `requirements.txt`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Model not loading | Check the path in the sidebar; verify checkpoint format |
| Random/garbage output | Checkpoint missing — app falls back to randomly initialized weights |
| Upload fails | Make sure `.streamlit/config.toml` exists and restart Streamlit |
| Slow inference | CPU-only by default; GPU is used automatically if available |

---

## Academic Context

- **Course:** Generative AI — AI4009
- **University:** FAST NUCES
- **Semester:** Spring 2026
- **Assignment:** 03
