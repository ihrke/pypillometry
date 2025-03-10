import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    requirements=fh.read().split()

with open("VERSION") as fh:
    version=fh.read().strip()
    
setuptools.setup(
    name="pypillometry",
    version=version,
    author="Matthias Mittner",
    author_email="matthias.mittner@uit.no",
    description="Pupillometry in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ihrke/pypillometry",
    packages=setuptools.find_namespace_packages(),#where="pypillometry", exclude="tests"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #include_package_data=True,
    package_dir={"pypillometry": "pypillometry"},
    package_data={
        "pypillometry": [],
        "pypillometry.data": ["*.asc"],
        "pypillometry.stan": ["*.stan"],
        },
    install_requires=requirements,
    python_requires='>=3.10',
)
