"""
https://github.com/Sofoclesias/music-source-separation
"""

from setuptools import setup, find_packages
import codecs

with open('requirements.txt',encoding='utf-16') as f:
    required = f.read().splitlines()

with codecs.open('README.md','r',encoding='utf-8') as f:
    readme = f.read()

setup(
    name='audiomancy',
    version="0.0.1",
    packages=find_packages(),
    description="Extractor de seis pistas instrumentales en canciones.",
    long_description=readme,
    install_requires=required
)