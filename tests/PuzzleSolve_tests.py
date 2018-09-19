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

#@nottest
def test_arrayMerge1():
    left = numpy.array([[1, 0], [1, 0], [1, 1]])
    right = numpy.array([[1, 1], [1,1]])

    leftCoord = (2, 1)
    rightCoord = (0, 0) 

    result  = solve._Cluster._prepArrays(left, right, leftCoord, rightCoord)

    nullRow = [0] * 4
    shouldLeft = numpy.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], nullRow])
    shouldRight = numpy.array([nullRow, nullRow, [0, 0, 1, 1], [0, 0, 1, 1]])

    #print(result[0])
    #print(result[1])

    numpy.testing.assert_array_equal(result[1], shouldRight)
    numpy.testing.assert_array_equal(result[0], shouldLeft)


def test_arrayMerge2():
    left = numpy.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]])
    right = numpy.array([[0, 0, 1], [1, 1, 1]])

    leftCoord = (1, 0)
    rightCoord = (1, 0)


    shouldLeft = numpy.array([[1, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    shouldRight = numpy.array([[0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 0]])

    result = solve._Cluster._prepArrays(left, right, leftCoord, rightCoord)

    numpy.testing.assert_array_equal(result[1], shouldRight)
    numpy.testing.assert_array_equal(result[0], shouldLeft)

def test_arrayMerge3():
    left = [
        [1, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 1]
    ]
    
    right = [
        [0, 0, 0],
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    leftArray = numpy.array(left)
    rightArray = numpy.array(right)

    leftCoord = (3, 3)
    rightCoord = (3, 0)

    shouldLeft  = [
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 0]
    ]

    shouldRight = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
    ]

    result = solve._Cluster._prepArrays(leftArray, rightArray, leftCoord, rightCoord)

    numpy.testing.assert_array_equal(result[1], shouldRight)
    numpy.testing.assert_array_equal(result[0] , shouldLeft)

def test_arrayMerge4():
    leftArray = numpy.array([[1,1,1], [1, 0, 1], [1, 1, 1]])
    rightArray = numpy.array([[1]])

    leftCoord = (1, 0)
    rightCoord = (0, 0)

    result = solve._Cluster._prepArrays(leftArray, rightArray, leftCoord, rightCoord)

    shouldLeft = numpy.copy(leftArray)
    shouldRight = numpy.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])


    numpy.testing.assert_array_equal(result[1], shouldRight)
    numpy.testing.assert_array_equal(result[0] , shouldLeft)
