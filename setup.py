from setuptools import setup, find_packages

# Read requirements.txt
def read_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()
    
setup(
    name="corde", 
    version="0.1.0",
    author="Sambit Sarkar",
    author_email="sambit.sarkar@tifr.res.in",
    description="A package regenerating lost samples in CORSIKA thinning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # For Markdown README
    url="https://github.com/oven8/corde",  # Your GitHub repository
    license="MIT",  # License for your package
    packages=find_packages(),  # Automatically finds subpackages
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Minimum Python version
)
