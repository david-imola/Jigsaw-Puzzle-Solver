from nose.tools import *
from PuzzleSolve.core import Puzzle
from PIL import Image

from PuzzleSolve.core import solve


import numpy
import os.path as path

pixels = 1500
inImage = path.join(path.dirname(__file__), "test.jpeg")
outImage = "/Volumes/Storage/local/puzzle/testpuzz.png"

#@nottest
def test_puzzle():
    with Puzzle() as puzz:
        puzz.create(inImage, outImage, pixels)

#@nottest
def test_solve():
    #inImage = path.join(path.dirname(__file__), "test.jpeg")
    #im = Image.open(inImage)
    #print("image")
    #print(numpy.asarray(im))

    solver = solve.JigsawTree(outImage, pixels)
    solver.solve()