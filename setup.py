from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="omniasr-headless",
    version="0.1.0",
    author="Bernard G. Tapiru, Jr.",
    author_email="tapirua@gmail.com",
    description="A headless adapter for Meta's Omnilingual ASR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/omniasr-headless",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "omnilingual-asr>=0.1.0",
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "soundfile>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "omniasr=omniasr_headless.cli:main",
        ],
    },
)
