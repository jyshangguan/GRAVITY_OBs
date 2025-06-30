from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gravity_obs",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Python package for preparing VLTI/GRAVITY observations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/user/GRAVITY_OBs",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "p2api": ["p2api"],
        "gspread": ["gspread", "oauth2client"],
        "dev": ["pytest", "black", "flake8", "sphinx"],
    },
    include_package_data=True,
    zip_safe=False,
)
