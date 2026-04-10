# GenAI Assignment 03

> A complete generative AI showcase built with PyTorch + Streamlit.
>
> Includes DCGAN, WGAN-GP, Pix2Pix, and CycleGAN with both standalone demos and one combined dashboard app.

## Overview

This project was developed for **Generative AI (AI4009)** at **FAST NUCES** (Spring 2026).

The repository demonstrates three practical image-generation and image-translation tasks:

- **Q1:** Compare DCGAN vs WGAN-GP and visualize mode collapse behavior.
- **Q2:** Translate sketch images to realistic/colorized outputs using Pix2Pix.
- **Q3:** Perform unpaired domain translation using CycleGAN (Sketch <-> Photo style).
- **Combined App:** Run all three questions from a single unified Streamlit UI.

## What Makes This Project Strong

- End-to-end interactive demos for all questions.
- Single combined app for easy presentation and grading.
- Supports local weight paths and direct upload of checkpoint files.
- Handles large model uploads via Streamlit server/client config.
- Includes both qualitative visualization and quantitative metrics (SSIM/PSNR in CycleGAN).

## Question-Wise Summary

| Question | Model            | Objective                                                  | Key Idea                                                        |
| -------- | ---------------- | ---------------------------------------------------------- | --------------------------------------------------------------- |
| Q1       | DCGAN vs WGAN-GP | Generate anime-style faces and compare diversity/stability | Wasserstein loss + gradient penalty improves training stability |
| Q2       | Pix2Pix (cGAN)   | Sketch -> realistic image translation                      | U-Net generator + PatchGAN discriminator + L1 reconstruction    |
| Q3       | CycleGAN         | Unpaired image-to-image translation                        | Cycle consistency enables translation without paired data       |

## Repository Structure

```text
.
|- App.py
|- app_q1.py
|- app_q2.py
|- app_q3.py
|- requirements.txt
|- .streamlit/config.toml
|- output/
|  |- dcgan_generator_final.pth
|  |- wgan_generator_final.pth
|  |- pix2pix_generator_final.pth
|  |- pix2pix_discriminator_final.pth
|- cyclegan_weights.pt
|- notebooks + report files
```

## Default Checkpoint Paths

The apps are configured with these defaults:

| Model               | Default Path                         |
| ------------------- | ------------------------------------ |
| DCGAN Generator     | `output/dcgan_generator_final.pth`   |
| WGAN-GP Generator   | `output/wgan_generator_final.pth`    |
| Pix2Pix Generator   | `output/pix2pix_generator_final.pth` |
| CycleGAN Checkpoint | `cyclegan_weights.pt`                |

If your files are in different locations, update paths in the sidebar or upload checkpoints from the UI.

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/Bilxl99/GenAI-Assignment03.git
cd GenAI-Assignment03
```

### 2) Create and activate virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Apps

### Combined Dashboard (recommended)

```bash
streamlit run App.py
```

### Standalone Demos

```bash
streamlit run app_q1.py
streamlit run app_q2.py
streamlit run app_q3.py
```

## Combined App Flow

When you launch `App.py`, the sidebar lets you switch between:

- **Home:** project overview and quick status.
- **Q1:** generate side-by-side DCGAN and WGAN-GP image grids.
- **Q2:** single or batch sketch translation with download support.
- **Q3:** bidirectional CycleGAN translation with optional cycle reconstruction and metrics.

## Streamlit Upload Configuration

Large model files are supported through `.streamlit/config.toml`:

- `server.maxUploadSize = 4096`
- `server.maxMessageSize = 4096`
- `client.maxUploadSize = 4096`

This is useful for `.pth` / `.pt` checkpoints that are too large for default limits.

## Requirements

Main dependencies:

- streamlit
- torch
- torchvision
- numpy
- Pillow
- scikit-image

All versions are listed in `requirements.txt`.

## Troubleshooting

- **Model not loading:** Verify path and checkpoint format from the sidebar status messages.
- **Random outputs:** Model file may be missing; app falls back to random-initialized weights.
- **Upload fails:** Ensure `.streamlit/config.toml` is present and restart Streamlit.
- **Slow inference:** Current setup runs on CPU by default for portability.

## Academic Context

- **Course:** Generative AI (AI4009)
- **University:** FAST NUCES
- **Semester:** Spring 2026
- **Assignment:** 03

## Contributors

- 22F-3845
- 22F-3360

---

If this repository helps your work, consider starring it on GitHub.
