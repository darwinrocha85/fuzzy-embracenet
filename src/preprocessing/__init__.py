"""
Preprocessing modules for extracting features from face, audio, and text modalities.
"""

from .video_vgg19 import VGG19Emotion, extraer_embedding_vgg19
from .audio_hubert import EmotionMLP, nested_loso_audio
from .text_dialogxl import DialogXLClassifier, nested_loso_texto
