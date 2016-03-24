# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='d606',
    version='0.0.1',
    description='d606 project',
    long_description=readme,
    author='M. Terndrup og jer andre (TODO: tilføj)',
    author_email='mternd13@student.aau.dk',
    url='https://github.com/mkote/dat6-d606-16',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
