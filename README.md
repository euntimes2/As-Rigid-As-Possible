# 🧩 As-Rigid-As-Possible (ARAP) Mesh Deformation — From Scratch

A clean and minimal implementation of the **ARAP deformation model**  
(as introduced by *Sorkine & Alexa, 2007*).

---

## ✨ Features

- 🔧 **Local Step** — SVD-based rotation estimation for each cell  
- 🔁 **Global Step** — Solve the Poisson linear system for vertex updates  
- 🎯 **Handle Constraints** — Arbitrary vertex positions can be fixed  
- 🔺 **Triangular Mesh Support** — Works on general triangle meshes  
- 🐍 **Pure Python Implementation** — Using NumPy + SciPy

---

## 📦 Installation

```bash
pip install -r requirements.txt
