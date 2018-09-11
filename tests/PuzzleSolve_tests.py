from nose.tools import *
from PuzzleSolve.core import Puzzle

from PuzzleSolve.core import solve


import numpy
import os.path as path

@nottest
def test_puzzle():

    pixels = 3000 #numpy.random.randint(200, 400)

    inImage = path.join(path.dirname(__file__), "test.jpeg")
    outImage = "/Volumes/Storage/local/puzzle/testpuzz.jpeg"


    with Puzzle() as puzz:
        puzz.create(inImage, outImage, pixels)


def test_solve():

    length = 500
    leftcol = numpy.random.randint(0, 255, (3, length))
    #rightcol = numpy.random.randint(0, 255, (3, length))
    leftcol2 = numpy.copy(leftcol)
    comp = solve.symetricCompatMeasure(leftcol, leftcol2)
    print("compat measure %r" % comp)
    assert False
