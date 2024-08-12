import os
import tempfile
import subprocess
import versioneer
import setuptools
from pathlib import Path
from setuptools.command.install import install

# -----------------------------------------------------------------------------

class GigapathInstall(install):
    """Installs gigapath, applying fix to broken pyproject.toml."""

    def run(self):
        if 'gigapath' in self.distribution.extras:
            self.install_gigapath()

        install.run(self)

    def install_gigapath(self):
        # Clone the gigapath repo
        repo_url = "git@github.com:prov-gigapath/prov-gigapath.git"
        clone_dir = Path(os.path.join(tempfile.gettempdir(), 'gigapath-repo'))

        if not clone_dir.exists():
            subprocess.run(["git", "clone", repo_url, clone_dir], check=True)

        # Fix the pyproject.toml file
        pyproject_file = clone_dir / "pyproject.toml"
        if pyproject_file.exists():
            with open(pyproject_file, "r") as file:
                content = file.read()

            # Modify the pyproject.toml content to include subdirectories
            new_content = content.replace(
                'include = ["gigapath"]',
                'include = ["gigapath", "gigapath.*"]'
            )

            with open(pyproject_file, "w") as file:
                file.write(new_content)

        # Install the modified gigapath package
        subprocess.run(["pip", "install", str(clone_dir)], check=True)

# -----------------------------------------------------------------------------

with open("README.md", "r") as fh:
    long_description = fh.read()

# -----------------------------------------------------------------------------

cmdclass = versioneer.get_cmdclass()
cmdclass.update({'install': GigapathInstall})

setuptools.setup(
    name="slideflow-noncommercial",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    author="James Dolezal",
    author_email="james@slideflow.ai",
    description="Non-commercial extensions and tools for Slideflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slideflow/slideflow-noncommercial",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License :: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'slideflow.plugins': [
            'extras = slideflow_noncommercial:register_extras',
        ],
    },
    python_requires='>=3.7',
    install_requires=[
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
            'fairscale'
        ]
    }
)
