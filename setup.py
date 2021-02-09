from distutils.core import setup


setup(
    name="surmod",
    packages=["surmod"],
    version="0.1.1",
    license="MIT",
    description="Surrogate modelling library for python including code for variable screening, sampling plans and models",
    author="Eniz Museljic",
    author_email="eniz.m@outlook.com",
    url="https://github.com/enizimus/surmod",
    download_url="https://github.com/enizimus/surmod/archive/v0.1.1-beta.tar.gz",
    keywords=["surogate", "models", "sampling plan", "screening"],
    install_requires=["numpy", "scipy", "matplotlib", "ypstruct", "torch"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)
