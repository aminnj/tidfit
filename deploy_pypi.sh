#!/usr/bin/env bash

# edit version in setup.py
rm -f dist/*
python3 setup.py bdist_wheel
python3 -m twine upload dist/*
# note, can do `pip install -e .` in this folder to install a "development" version of the package
