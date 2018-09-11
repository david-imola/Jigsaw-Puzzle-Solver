from PIL import Image
import numpy as np
from enum import Enum
import heapq


def _addRandomNoise(array):
    np.random.seed(0)  # We actually want consistent noise
    noisey = np.random.randint(0, 2, size=(3, 9))
    return np.concatenate((array, noisey), 1)


#TODO FIx this. currently, it's coded to find the mean of
#one column (silly goose). It was supposed to do it between 
#The leftmost and leftmost one next to it
def _getInverseMeanAndCovariance(gradArray):
    mean = np.mean(gradArray, 1)
    noiseyGrad = _addRandomNoise(gradArray)
    covInv = np.linalg.inv(np.cov(noiseyGrad))
    return (mean, covInv)


def _compatMeasure(grad, mean, covInv):
    score = float(0)
    P = grad.shape[0]
    for p in range(P):
        deviation = np.transpose(grad[:, p]) - mean
        term = np.matmul(deviation, covInv)
        score += np.matmul(term, deviation)
    return score


def symetricCompatMeasure(leftColumn, rightColumn):
    l_mean, l_covInv = _getInverseMeanAndCovariance(leftColumn)
    r_mean, r_covInv = _getInverseMeanAndCovariance(rightColumn)

    grad_lr = np.subtract(rightColumn, leftColumn)
    grad_rl = np.multiply(grad_lr, -1)

    d_lr = _compatMeasure(grad_lr, l_mean, l_covInv)
    d_rl = _compatMeasure(grad_rl, r_mean, r_covInv)

    return d_lr + d_rl


class _Piece:
    def __init__(self, imgArray, i, j, pieceLen):
        self.i, self.j = i, j
        x0, y0 = j * pieceLen, i * pieceLen
        x1, y1 = x0 + pieceLen, y0 + pieceLen
        pieceArray = imgArray[x0:x1, y0:y1]
        # 0 for top, 1 for right, 2 for bottom, 3 for left
        #TODO fix this. This should store the absolute edges,
        #not the gradient along the edges. The thing that should
        #be stored in that regrad is the mean and the covariance 
        #matrix of the gradient
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
    def __init__(self, piece1, piece2, orient1, orient2):
        self.pieces = (piece1, piece2)
        self.orientation = orient1 * 4 + orient2
        self.dissim = symetricCompatMeasure(piece1[orient1], piece2[orient2])

    def __lt__(self, other):  # to get heapq to work with this
        return self.dissim < other.dissim

    def __eq__(self, other):
        return self.dissim == self.dissim

    def sift(self, secondSmallest):
        self.dissim = self.dissim / secondSmallest

    @staticmethod
    def addEdgesToHeap(heap, piece1, piece2):
        edges = []
        for i in range(4):
            for j in range(4):
                newEdge = _Edge(piece1, piece2, i, j)
                edges.append(newEdge)
        edges = sorted(edges)
        min2 = edges[1].dissim
        for edge in edges:
            edge.sift(min2)
            heapq.heappush(heap, edge)

    @staticmethod
    def getOrientations(orientation):
        orient1 = orientation / 4
        orient2 = orientation % 4
        return (orient1, orient2)


class JigsawTree:
    def __init__(self, inFilename, pieceLen):
        imIn = Image.open(inFilename)
        rows = int(imIn.size[0]/pieceLen)
        cols = int(imIn.size[1]/pieceLen)

        imArray = np.asarray(imIn)

        self.edges = [] # TODO doa a priority/heap queue here (heapq)
        pieces = []
        for i in range(rows):
            for j in range(cols):
                newPiece = _Piece(imArray, i, j, pieceLen)
                for piece in self.pieces:
                    _Edge.addEdgesToHeap(self.edges, piece, newPiece)
                pieces.append(newPiece)