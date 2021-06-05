#!/usr/bin/env python
from setuptools import setup


with open("./README.md") as f:
    long_description = f.read()

setup(name='amtokenizers',
      version='0.0.9',
      description='Amharic language tokenizers',
      author='Mitiku Yohannes',
      author_email='se.mitiku.yohannes@gmail.com',
      packages=["amtokenizers"],
      install_requires = ["transformers"],
      license="MIT",
      long_description_content_type="text/markdown",
      long_description = long_description,
      include_package_data=True,
      package_data = {
          "":["data/*"]
      }
      
)