from setuptools import setup, find_packages

setup(
    name="TheAmazingProc",
    url="https://github.com/andnp/TheAmazingProc.git",
    author="Andy Patterson",
    author_email="andnpatterson@gmail.com",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["numba", "numpy"],
    version='0.1.0',
    license="MIT",
    description="",
    long_description="todo",
    extras_require={
        "dev": [
            "mypy",
            "flake8",
            "commitizen",
            "pre-commit",
            "pipenv-setup[black]",
        ]
    },
)
