# GenAI Assignment 03 — GANs for Image Synthesis

**Course:** Generative AI (AI4009) | Spring 2026  
**University:** FAST National University of Computer and Emerging Sciences  
**Team Members:** Rana M Ahmad(22F-8758) & Urwa Sajid(22F-3244)

---

## 📌 Overview

This repository contains implementations of three GAN-based image synthesis systems built in PyTorch and deployed via Streamlit.

| Question | Model | Task |
|----------|-------|------|
| Q1 | DCGAN + WGAN-GP | Tackling Mode Collapse |
| Q2 | Pix2Pix | Sketch → Realistic Image Translation |
| Q3 | CycleGAN | Unpaired Domain Adaptation |

---

## 🗂️ Repository Structure
├── App.py                  # Main Streamlit app (all 3 questions)
├── app_q1.py               # Q1 standalone app
├── app_q2.py               # Q2 standalone app
├── app_q3.py               # Q3 standalone app
├── Q1_DCGAN_WGANGP_FIXED.ipynb
├── Q2_Pix2Pix.ipynb
├── Q3_CycleGAN.ipynb
├── requirements.txt
└── .streamlit/
└── config.toml

---

## ❓ Question 1: DCGAN vs WGAN-GP (Mode Collapse)

- **Dataset:** Pokemon Sprites + Anime Faces (64×64)
- **Baseline:** DCGAN with BCE Loss
- **Improved:** WGAN-GP with Wasserstein Loss + Gradient Penalty (λ=10)
- **Key Result:** WGAN-GP eliminates mode collapse and produces more diverse samples

## ❓ Question 2: Pix2Pix (Paired Image Translation)

- **Dataset:** CUHK Face Sketch + Anime Sketch Colorization
- **Model:** U-Net Generator + PatchGAN Discriminator
- **Loss:** Adversarial Loss + L1 Reconstruction Loss
- **Metrics:** SSIM, PSNR

## ❓ Question 3: CycleGAN (Unpaired Domain Translation)

- **Dataset:** TU-Berlin Sketch + Sketchy Dataset + Google QuickDraw
- **Model:** ResNet-based Generators (G_AB, G_BA) + PatchGAN Discriminators
- **Loss:** Adversarial + Cycle Consistency + Identity Loss
- **Metrics:** SSIM, PSNR

---

## 🚀 Running the App

```bash
pip install -r requirements.txt
streamlit run App.py
```

---
