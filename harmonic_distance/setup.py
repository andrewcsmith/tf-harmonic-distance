import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="harmonic_distnace",
    version="0.0.1",
    author="Andrew C. Smith",
    author_email="andrewchristophersmith@gmail.com",
    description="Tenney's harmonic distance metrics in tensorflow",
    long_description=long_description,
    url="https://github.com/andrewcsmith/tf-harmonic-distance",
    packages=setuptools.find_packages(),
    classifiers=[
        "Porgramming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)