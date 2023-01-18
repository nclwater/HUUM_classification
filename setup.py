import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="huum_classification-SBerendsen", # Replace with your own username
    version="1.0.0",
    author="Sven Berendsen",
    author_email="s.berendsen2@newcastle.ac.uk",
    description="Classifies a given time series diurnal pattern",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/LeoTolstoi/huum_classification/",
    project_urls={
        "Bug Tracker": "https://gitlab.com/LeoTolstoi/huum_classification/-/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    # packages=['matplotlib', 'numpy', 'pandas', 'seaborn', 'plotly', 'kaleido-python'],
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
