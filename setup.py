#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
from setuptools import setup, find_packages


requirements = ["scikit-learn", "python-crfsuite"]

if sys.version_info[:2] < (2, 7):
    requirements.append('argparse')
if sys.version_info[:2] < (3, 4):
    requirements.append('enum34')
if sys.version_info[:2] < (3, 5):
    requirements.append('typing')

extras_require = {
    ':python_version<"2.7"': ['argparse'],
    ':python_version<"3.4"': ['enum34'],
    ':python_version<"3.5"': ['typing'],
}

setup(
    name="jiayan",
    version="0.0.21",
    author="Jiajie Yan",
    author_email="jiaeyan@gmail.com",
    description="The NLP toolkit designed for classical chinese.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    license="MIT",
    url="https://github.com/jiaeyan/Jiayan",
    keywords=['classical-chinese', 'ancient-chinese', 'nlp'],
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    python_requires='>=2.6, >=3',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities',
        'Topic :: Text Processing',
    ]
)
