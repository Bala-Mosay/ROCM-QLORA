"""
Setup script for rocm-qlora package.
"""

from setuptools import setup, find_packages

setup(
    name="rocm-qlora",
    version="0.1.0",
    description="A pure PyTorch, ROCm-native QLoRA fine-tuning library for AMD GPUs.",
    author="Antigravity",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    python_requires=">=3.9",
)
