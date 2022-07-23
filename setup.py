from setuptools import setup
import climl


setup(
    name="climl",
    packages=["climl", "climl.utils"],
    install_requires=[
        "pandas >= 1.4.2",
        "scikit-learn >= 1.0.2",
        "numpy >= 1.22.3",
        "fire >= 0.4.0",
    ],
    python_requires=">=3.8.0",
    entry_points={"console_scripts": ["climl = climl.app:main"]},
    version=climl.__version__,
    description="A test package",
    author="rocanale",
    author_email="",
    url="",
)
