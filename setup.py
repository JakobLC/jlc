from setuptools import setup, find_packages


setup(
    name="jlc",
    author='Jakob Loenborg Christensen',
    author_email='jakoblc@live.dk',
    description="Useful functions.",
    version="1.9.0",
    packages=find_packages(),
    install_requires=["tifffile",
                      "numpy",
                      "scipy",
                      "matplotlib",
                      "torch"]
)