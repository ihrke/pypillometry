import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pypillometry",
    version="1.0.1",
    author="Matthias Mittner",
    author_email="matthias.mittner@uit.no",
    description="Pupillometry in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ihrke/pypillometry",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
