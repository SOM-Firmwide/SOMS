#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Ricardo Avelino",
    author_email='ricardo.avelino@som.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Repository for structural analysis and design",
    entry_points={
        'console_scripts': [
            'soms=soms.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='soms',
    name='soms',
    packages=find_packages(include=['soms', 'soms.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ricardoavelino2/soms',
    version='0.1.0',
    zip_safe=False,
)
