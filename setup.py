from setuptools import find_packages, setup

REQUIREMENTS = [
    i.strip() for i in open("requirements.txt").readlines() if not i.startswith("--")
]

setup(
    name="Snake",
    version="0.1.0",
    url="git@github.com:jofa974/Snake.git",
    author="Jonathan Faustin",
    author_email="faustin.jonathan@gmail.com",
    description="The game of Snake solved by algorithms",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
)
