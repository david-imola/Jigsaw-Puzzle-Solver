from PIL import Image
import numpy as np
from enum import Enum


def _addRandomNoise(array):
    np.random.seed(0)  # We actually want consistent noise
    noisey = np.random.randint(0, 2, size=(3, 9))
    return np.concatenate(array, noisey)


def _getInverseMeanAndCovariance(gradArray):
    mean = np.mean(gradArray, axis=0)
    noiseyGrad = _addRandomNoise(gradArray)
    covInv = np.linalg.inv(np.cov(noiseyGrad))
    return (mean, covInv)


def _compatMeasure(grad, mean, covInv):
    score = 0
    P = leftArray.shape[0]
    for p in range(P):
        deviation = np.subtract(np.transpose(noiseyGrad), mean)
        term = np.matmul(deviation, covInv)
        score += np.matmul(term, deviation)
    return score


def _symetricCompatMeasure(leftColumn, rightColumn):
    (leftMean, leftInvvariance) = _getInverseMeanAndCovariance(leftColumn)
    (rightMean,)

class _Piece:
    def __init__(self, imgArray, i, j, pieceLen):
        self.i, self.j = i, j
        x0, y0 = j * pieceLen, i * pieceLen
        x1, y1 = x0 + pieceLen, y0 + pieceLen
        pieceArray = imgArray[x0:x1, y0:y1]
         # 0 for top, 1 for right, 2 for bottom, 3 for left
        left = np.subtract(pieceArray[:, 1], pieceArray[:, 0])
        right = np.subtract(pieceArray[:, pieceLen - 2],
                                  pieceArray[:, pieceLen - 1])
        bottom = np.subtract(pieceArray[pieceArray - 2, :],
                                   pieceArray[pieceArray - 1, :])
        top = np.subtract(pieceArray[1, :], pieceArray[0, :])
        self.grad = (top, right, bottom, left)
        for i in range(4):
            self.grad[i] = np.reshape(self.grad[i], newshape=(3, pieceLen))


class _Edge:
    def __init__(self, piece0, piece1, orient0, orient1):
        self.pieces = (piece0, piece1)
        self.orientation = (orient0, orient1)
        # TODO implement cost function here, so that it can determine the orientation

    def __lt__(self, other): #to get heapq to work with this
        pass
    def __eq__(self, other):
        pass


def _getAllEdges(piece0, piece1)
    edges = []
    for i in range(4):
        for j in range(4):
            newEdge = _Edge(piece0, piece1, i, j)


class JigsawTree:
    def __init__(self, inFilename, pieceLen):
        imIn = Image.open(inFilename)
        rows = int(imIn.size[0]/pieceLen)
        cols = int(imIn.size[1]/pieceLen)

        imArray = numpy.asarray(imIn)

        self.edges = [] # TODO doa a priority/heap queue here (heapq)
        self.pieces = []
        for i in range(rows):
            for j in range(cols):
                newPiece = _Piece(imArray, i, j, pieceLen)
                for piece in self.pieces

