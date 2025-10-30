from setuptools import setup, find_packages

setup(
    name="if97_py",
    version="1.0.0",
    url="https://github.com/Sam-Strand/if97_py",
    author="Садовский М.К.",
    author_email="i@maxim-sadovskiy.ru",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.55.0"
    ]
)
