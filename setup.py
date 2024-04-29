import setuptools

from src.version import __version__

setuptools.setup(
    name="LMS",
    version=__version__,
    author="IAIS",
    description="Language Model Service",
    packages=setuptools.find_packages(exclude=["src*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    install_requires=["pydantic==2.5.1", "requests>=2.27.1"],
)
