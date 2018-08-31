
from setuptools import setup

#get version from version file

with open('VERSION', 'r') as version_file:
    version = version_file.read().strip()


config = {
    'description': 'Creates and solves a jigsaw puzzle from an image,'
    ' where the pieces are all squares.',
    'author': 'David Imola',
    'url': "Link to project's home page",
    'author_email': 'imola.david72@gmail.com',
    'version': version,
    'install_requires': ['wxPython', 'scipy', 'Pillow'],
    'packages': ['puzzlesolve'],
    'name': 'projectname'
}

setup(**config)
