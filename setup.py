from setuptools import setup, find_packages

setup(
    name="ai-drug-repo",
    version="2.0.0",
    author="dino65-dev",
    description="The world's first pharmacopoeia for large language models.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "transformer_lens>=1.12.0",
        "sae-lens>=3.0.0",
        "repeng>=0.1.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "jupyter>=1.0.0"],
    },
)
