import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dynamapp",
    version="1.0.0",
    author="Wissem Chiha",
    author_email="chihawissem@gmail.com",
    description="A differentiable package for \
        representation and identification of multibody dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wissem01chiha/dynamapp",
    packages=setuptools.find_packages()
    )
