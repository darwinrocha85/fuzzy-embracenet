# FuzzyEmbraceNet: Neuro-fuzzy Integration for Multimodal Emotion Recognition

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/EmbraceNet_Fuzzy_2026.pdf)

**[English](#english) | [Español](#español)**

</div>

---

<a name="english"></a>

## 🇬🇧 English

### Overview

**FuzzyEmbraceNet** is a neuro-fuzzy multimodal architecture for emotion recognition that extends the [EmbraceNet](https://doi.org/10.1016/j.inffus.2019.02.010) framework by incorporating **Gaussian fuzzy activation functions** and adaptive fusion mechanisms. This design enables the model to better represent uncertainty and treat affect as a continuous rather than strictly discrete phenomenon.

### 🎯 Key Features

- **Multimodal Fusion**: Integrates face, audio, and text modalities for robust emotion recognition
- **Fuzzy Logic Integration**: Gaussian membership functions model emotional ambiguity
- **Robust to Missing Modalities**: Maintains performance even when one modality is unavailable
- **Nested LOSO Protocol**: Leak-free evaluation ensuring no data contamination across folds
- **State-of-the-art Results**: 87.03% accuracy and 86.52% F1-score on IEMOCAP

### 📊 Results

Evaluated on the **IEMOCAP** dataset (4 emotions: anger, happiness, sadness, neutral):

| Metric          | FuzzyEmbraceNet | EmbraceNet (ReLU) | Improvement |
| --------------- | --------------- | ----------------- | ----------- |
| **Accuracy**    | 87.03%          | 77.00%            | +10.03 pp   |
| **F1-Macro**    | 86.52%          | 78.00%            | +8.52 pp    |
| **F1-Weighted** | 87.02%          | 78.70%            | +8.32 pp    |

**Per-class F1 scores:**

- Anger: 0.84
- Happiness: 0.90
- Sadness: 0.86
- Neutral: 0.82

### 🏗️ Architecture

FuzzyEmbraceNet processes three modalities independently before fusion:

1. **Face (Visual)**: VGG19 pretrained on ImageNet with Haar Cascade face detection
   - Input: 8 frames per utterance (48×48 pixels)
   - Standalone accuracy: 41%

2. **Audio**: HuBERT + 3-layer MLP (nested LOSO)
   - Embeddings: Mean of last 4 hidden states (1024-dim)
   - Standalone accuracy: 65%

3. **Text**: DialogXL (XLNet-base + Transformer encoder)
   - Max tokens: 2048
   - Standalone accuracy: 79%

**Fusion**: Probabilistic component-wise selection with **Gaussian fuzzy activation** in docking layers.

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/darwinrocha85/fuzzy-embracenet.git
cd fuzzy-embracenet

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🚀 Quick Start

#### 1. Download IEMOCAP Dataset

The IEMOCAP dataset is required. You can access it from:

- **Kaggle**: [IEMOCAP Dataset](https://www.kaggle.com/datasets/jiten597/iemocap)
- **Official**: [USC IEMOCAP](https://sail.usc.edu/iemocap/)

#### 2. Run Full Pipeline

```bash
# Set IEMOCAP path in src/config.py or via environment variable
export IEMOCAP_ROOT="/path/to/IEMOCAP_full_release"

# Run preprocessing and training (sequential execution)
python -c "
exec(open('src/config.py').read())
exec(open('src/preprocessing/video_vgg19.py').read())
exec(open('src/preprocessing/audio_hubert.py').read())
exec(open('src/preprocessing/text_dialogxl.py').read())
exec(open('src/fusion/fusion_pipeline.py').read())
"
```

#### 3. Use Individual Components

```python
from src.config import *
from src.models.embrace_net_fuzzy import EmbraceNetFuzzy

# Load pretrained model (after training)
model = EmbraceNetFuzzy(K=32).to(DEVICE)
# ... inference code
```

### 📂 Repository Structure

```
fuzzy-embracenet/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── src/                               # Source code
│   ├── config.py                      # Global configuration
│   ├── preprocessing/                 # Modality preprocessing
│   │   ├── video_vgg19.py            # Face embeddings (VGG19)
│   │   ├── audio_hubert.py           # Audio embeddings (HuBERT)
│   │   └── text_dialogxl.py          # Text embeddings (DialogXL)
│   ├── models/                        # Neural architectures
│   │   ├── embrace_net_fuzzy.py      # FuzzyEmbraceNet model
│   │   └── mlp_classifier.py         # Audio MLP classifier
│   └── fusion/                        # Multimodal fusion
│       └── fusion_pipeline.py        # Late fusion + grid search

```

### 🔬 Methodology

**Nested Leave-One-Session-Out (LOSO)**:

- For each outer fold K (test session):
  - For each inner session J:
    - If J == K: train on {1..5} \ {K}, test on K
    - If J ≠ K: train on {1..5} \ {K, J}, test on J
- **15 unique training configurations** ensuring zero data leakage

**Training Details**:

- Optimizer: Adam (lr=3×10⁻⁴)
- Loss: Cross-Entropy
- Early stopping: patience=20 epochs
- Hardware: 2× NVIDIA Tesla T4 (16GB VRAM each) on Kaggle

### 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{rocha2026fuzzy,
  title={Fuzzy EmbraceNet: Neuro-fuzzy integration for multimodal emotion recognition},
  author={Rocha, Darwin and Carrasquel, Soraya and Coronado, David and Aguilera, Ana},
  journal={[Journal Name]},
  year={2026},
  note={Manuscript submitted for publication}
}
```

### 📧 Contact

- **Darwin Rocha**: darwinrocha85@gmail.com
- **Ana Aguilera**: ana.aguilera@uv.cl
- **Soraya Carrasquel**: scarrasquel@usb.ve
- **David Coronado**: dcoronado@usb.ve

### 🙏 Acknowledgments

- IEMOCAP dataset creators (USC SAIL)
- Kaggle for providing free GPU resources
- Original EmbraceNet authors: [Choi & Lee (2019)](https://doi.org/10.1016/j.inffus.2019.02.010)
- HuggingFace for pretrained models

### 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<a name="español"></a>

## 🇪🇸 Español

### Descripción General

**FuzzyEmbraceNet** es una arquitectura neuro-difusa multimodal para reconocimiento de emociones que extiende el marco [EmbraceNet](https://doi.org/10.1016/j.inffus.2019.02.010) incorporando **funciones de activación difusas Gaussianas** y mecanismos de fusión adaptativa. Este diseño permite al modelo representar mejor la incertidumbre y tratar el afecto como un fenómeno continuo en lugar de estrictamente discreto.

### 🎯 Características Principales

- **Fusión Multimodal**: Integra modalidades faciales, de audio y texto para un reconocimiento robusto de emociones
- **Integración de Lógica Difusa**: Las funciones de membresía Gaussiana modelan la ambigüedad emocional
- **Robusto ante Modalidades Faltantes**: Mantiene el rendimiento incluso cuando una modalidad no está disponible
- **Protocolo Nested LOSO**: Evaluación sin fugas de datos garantizando que no haya contaminación entre folds
- **Resultados de Vanguardia**: 87.03% de accuracy y 86.52% de F1-score en IEMOCAP

### 📊 Resultados

Evaluado en el dataset **IEMOCAP** (4 emociones: anger, happiness, sadness, neutral):

| Métrica         | FuzzyEmbraceNet | EmbraceNet (ReLU) | Mejora    |
| --------------- | --------------- | ----------------- | --------- |
| **Accuracy**    | 87.03%          | 77.00%            | +10.03 pp |
| **F1-Macro**    | 86.52%          | 78.00%            | +8.52 pp  |
| **F1-Weighted** | 87.02%          | 78.70%            | +8.32 pp  |

**F1-scores por clase:**

- Anger (Ira): 0.84
- Happiness (Felicidad): 0.90
- Sadness (Tristeza): 0.86
- Neutral: 0.82

### 🏗️ Arquitectura

FuzzyEmbraceNet procesa tres modalidades de forma independiente antes de la fusión:

1. **Face (Visual)**: VGG19 preentrenado en ImageNet con detección de rostros Haar Cascade
   - Entrada: 8 frames por utterance (48×48 píxeles)
   - Accuracy standalone: 41%

2. **Audio**: HuBERT + MLP de 3 capas (nested LOSO)
   - Embeddings: Media de los últimos 4 hidden states (1024-dim)
   - Accuracy standalone: 65%

3. **Texto**: DialogXL (XLNet-base + Transformer encoder)
   - Tokens máximos: 2048
   - Accuracy standalone: 79%

**Fusión**: Selección probabilística componente a componente con **activación difusa Gaussiana** en las capas de acoplamiento.

### 📦 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/darwinrocha85/fuzzy-embracenet.git
cd fuzzy-embracenet

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 🚀 Inicio Rápido

#### 1. Descargar el Dataset IEMOCAP

Se requiere el dataset IEMOCAP. Puedes acceder desde:

- **Kaggle**: [IEMOCAP Dataset](https://www.kaggle.com/datasets/jiten597/iemocap)
- **Oficial**: [USC IEMOCAP](https://sail.usc.edu/iemocap/)

#### 2. Ejecutar Pipeline Completo

```bash
# Configurar ruta de IEMOCAP en src/config.py o vía variable de entorno
export IEMOCAP_ROOT="/ruta/a/IEMOCAP_full_release"

# Ejecutar preprocesamiento y entrenamiento (ejecución secuencial)
python -c "
exec(open('src/config.py').read())
exec(open('src/preprocessing/video_vgg19.py').read())
exec(open('src/preprocessing/audio_hubert.py').read())
exec(open('src/preprocessing/text_dialogxl.py').read())
exec(open('src/fusion/fusion_pipeline.py').read())
"
```

### 🔬 Metodología

**Nested Leave-One-Session-Out (LOSO)**:

- Para cada fold externo K (sesión de test):
  - Para cada sesión interna J:
    - Si J == K: entrenar en {1..5} \ {K}, testear en K
    - Si J ≠ K: entrenar en {1..5} \ {K, J}, testear en J
- **15 configuraciones únicas de entrenamiento** garantizando cero fuga de datos

**Detalles de Entrenamiento**:

- Optimizador: Adam (lr=3×10⁻⁴)
- Pérdida: Cross-Entropy
- Early stopping: paciencia=20 épocas
- Hardware: 2× NVIDIA Tesla T4 (16GB VRAM cada una) en Kaggle

### 📄 Citar Este Trabajo

Si usas este código en tu investigación, por favor cita:

```bibtex
@article{rocha2026fuzzy,
  title={Fuzzy EmbraceNet: Neuro-fuzzy integration for multimodal emotion recognition},
  author={Rocha, Darwin and Carrasquel, Soraya and Coronado, David and Aguilera, Ana},
  journal={[Nombre del Journal]},
  year={2026},
  note={Manuscrito enviado para publicación}
}
```

### 📧 Contacto

- **Darwin Rocha**: darwinrocha85@gmail.com
- **Ana Aguilera**: ana.aguilera@uv.cl
- **Soraya Carrasquel**: scarrasquel@usb.ve
- **David Coronado**: dcoronado@usb.ve

### 📜 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

<div align="center">

**⭐ If you find this work useful, please consider starring the repository! ⭐**

**⭐ Si este trabajo te resulta útil, ¡considera darle una estrella al repositorio! ⭐**

</div>
