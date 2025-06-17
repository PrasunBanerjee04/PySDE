from setuptools import setup, find_packages

setup(
    name="fin-sde",
    version="0.1.0",
    description="Simulation and estimation toolkit for financial stochastic differential equations",
    author="Prasun Banerjee",
    author_email="prasun.banerjee04@gmail.com",
    url="https://github.com/PrasunBanerjee04/PySDE",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
        "numba",
        "torch"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Mathematical Modeling :: Mathematics"
    ],
    python_requires='>=3.7',
)
