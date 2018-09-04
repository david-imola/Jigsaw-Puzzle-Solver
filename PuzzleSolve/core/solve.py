from PIL import Image
import numpy as np


def _mahaDistanceSymetric():
    pass


class _Piece:
    def __init__(self, imgArray, i, j, pieceLen):
        self.i, self.j = i, j
        x0, y0 = j * pieceLen, i * pieceLen
        x1, y1 = x0 + pieceLen, y0 + pieceLen
        pieceArray = imgArray[x0:x1, y0:y1]
        self.grad_left = pieceArray[:, 1] - pieceArray[:, 0]
        self.grad_right = pieceArray[:, pieceLen - 2]\
            - pieceArray[:, pieceLen - 1]
        self.grad_bottom = pieceArray[pieceArray - 2, :]\
            - pieceArray[pieceArray - 1, :]
        self.grad_top = pieceArray[1, :] - pieceArray[0, :]


class _Edge:
    def __init__(self, piece0, piece1):
        self.piece0 = piece0
        self.piece1 = piece1
        # TODO implement cost function here, so that it can determine the orientation


class JigsawTree:
    def __init__(self, inFilename, pieceLen):
        imIn = Image.open(inFilename)
        rows = int(imIn.size[0]/pieceLen)
        cols = int(imIn.size[1]/pieceLen)

        imArray = numpy.asarray(imIn)

        self.pieces = []
        for i in range(rows):
            for j in range(cols):
                self.pieces.append(_Piece(imArray, i, j, pieceLen))
