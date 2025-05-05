from pathlib import Path
from setuptools import setup, find_packages


with open("README.md") as f:
    long_description = f.read()
    
current_path = Path(__file__).parent

with open('README.md') as f:
    long_description = f.read()

setup(
    name="moviad",
    version="0.4",
    author="AMCO-UNIPD",
    author_email="your@email.com",
    description="A modular and lightweight framework for Visual Anomaly Detection.",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        # local requirements
        f"mcunet @ {Path(current_path, 'moviad', 'backbones', 'mcunet').resolve().as_uri()}",
        # external requirements
        "opencv-python",
    ],
)
