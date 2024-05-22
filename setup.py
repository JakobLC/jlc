from setuptools import setup, find_packages


setup(
    name="jlc",
    author='Jakob Loenborg Christensen',
    author_email='jloch@dtu.dk',
    description="Useful functions for image processing.",
    version="2.0.1",
    packages=find_packages(),
    install_requires=["tifffile",
                      "numpy",
                      "scipy",
                      "matplotlib",
                      "torch"]
)