"""
https://github.com/Sofoclesias/music-source-separation
"""

from setuptools import setup
import codecs

with codecs.open('README.md','r',encoding='utf-8') as f:
    readme = f.read()

setup(
    name='audiomancy',
    version="0.0.1",
    description="Extractor de seis pistas instrumentales en canciones.",
    long_description=readme,
)