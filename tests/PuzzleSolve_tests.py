from nose.tools import *
from PuzzleSolve.core import Puzzle

import numpy
import os.path as path


def test_puzzle():

    pixels = 3000 #numpy.random.randint(200, 400)

    inImage = path.join(path.dirname(__file__), "test.jpeg")
    outImage = "/Volumes/Storage/local/puzzle/testpuzz.jpeg"


    with Puzzle() as puzz:
        puzz.create(inImage, outImage, pixels)
