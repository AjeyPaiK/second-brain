from setuptools import setup, find_packages

setup(
    name="second-brain",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "rich>=13.0.0",
        "httpx>=0.24.0",
    ],
    entry_points={
        "console_scripts": [
            "second-brain = cli.main:cli",
        ],
    },
    python_requires=">=3.10",
)
