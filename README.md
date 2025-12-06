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
```

## 🎥 Visualization (Linux/WSL Friendly)

The deformation result is rendered with **Plotly** so you can open it in any
web browser—even when running the computation headlessly on Linux/WSL.

1. Run `python arap_main.py` to perform the deformation.
2. The script saves an interactive HTML file (default: `arap_deformation.html`).
3. Open the HTML in your browser to inspect and rotate the mesh interactively.
