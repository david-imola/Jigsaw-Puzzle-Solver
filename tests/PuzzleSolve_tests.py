from nose.tools import *
from PuzzleSolve.core import Puzzle
from PIL import Image

from PuzzleSolve.core import solve


import numpy
import os.path as path

pixels = 1500
inImage = path.join(path.dirname(__file__), "test.jpeg")
outImage = "/Volumes/Storage/local/puzzle/testpuzz.png"

@nottest
def test_puzzle():
    with Puzzle() as puzz:
        puzz.create(inImage, outImage, pixels)

@nottest
def test_solve():
    #inImage = path.join(path.dirname(__file__), "test.jpeg")
    #im = Image.open(inImage)
    #print("image")
    #print(numpy.asarray(im))

    solver = solve.JigsawTree(outImage, pixels)
    solver.solve()

def test_arrayMerge():
    left = numpy.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]])
    right = numpy.array([[1, 1], [1,1]])

    leftCoord = (2, 1)
    rightCoord = (0, 0) 

    result  = solve._Cluster._prepArrays(left, right, leftCoord, rightCoord)

    nullRow = [0] * 5
    shouldLeft = numpy.array([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], nullRow])
    shouldRight = numpy.array([nullRow, nullRow, [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])

    print(result[0])
    print(result[1])

    numpy.testing.assert_array_equal(result[1], shouldRight)
    numpy.testing.assert_array_equal(result[0], shouldLeft)