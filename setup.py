from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="radiotherapy-env",
    version="1.0.0",
    author="Vaishnavi Agrawal",
    author_email="vagrawal_be22@thapar.edu",
    description="OpenEnv RL environment for radiotherapy treatment planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VaishnaviAgrawal03/radio_therapy_planning_for_tumor_treatment",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "Pillow>=9.0.0",
        "scikit-image>=0.20.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "training": [
            "stable-baselines3>=2.0.0",
            "torch>=2.0.0",
        ],
        "inference": ["openai>=1.0.0"],
        "demo": ["gradio>=4.0.0"],
        "dev":  ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "radiotherapy-train=baseline.train_ppo:main",
            "radiotherapy-eval=baseline.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="reinforcement-learning gymnasium radiotherapy cancer openenv",
)
