import os
import tempfile
import subprocess
import versioneer
import setuptools
from pathlib import Path
from setuptools.command.install import install

# -----------------------------------------------------------------------------

with open("README.md", "r") as fh:
    long_description = fh.read()

# -----------------------------------------------------------------------------

setuptools.setup(
    name="slideflow-noncommercial",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="James Dolezal",
    author_email="james@slideflow.ai",
    description="Non-commercial extensions and tools for Slideflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slideflow/slideflow-noncommercial",
    packages=setuptools.find_packages(),
    license="Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'slideflow.plugins': [
            'extras = slideflow_noncommercial:register_extras',
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        'slideflow>=3.0',
        'numpy',
        'pillow>=6.0.0',
        'gdown',
        'torch',
        'torchvision',
        'timm',
        'huggingface_hub',
        'transformers',
        'fastai',
        'scikit-misc'
    ],
    extras_require={
        'gigapath': [
            'timm>=1.0.3',
            'fairscale',
            'flash_attn==2.5.8',
            'einops'
        ]
    }
)
