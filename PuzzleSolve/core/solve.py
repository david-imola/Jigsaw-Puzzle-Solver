from PIL import Image
import numpy as np
from enum import Enum
import heapq


def _addRandomNoise(array):
    np.random.seed(0)  # We actually want consistent noise
    noisey = np.random.randint(0, 2, size=(3, 9))
    return np.concatenate((array, noisey), 1)


def _getInverseCovariance(gradArray):
    noiseyGrad = _addRandomNoise(gradArray)
    return np.linalg.inv(np.cov(noiseyGrad))


def _compatMeasure(grad, mean, covInv):
    score = float(0)
    P = grad.shape[0]
    #printe = True
    for p in range(P):
        #row = np.absolute(grad[p, :])
        deviation = grad[p, :] - mean
        #if printe and np.any(np.less([10, 10, 10], row)):
        #    printe = False
        term = np.matmul(deviation, covInv)
        score += np.matmul(term, deviation)
    #if(printe):
    #    print(score)
    return score#, printe

#TODO Not working. edge weights must be wrong or something

class _Piece:
    def __init__(self, imgArray, i, j, pieceLen):


        self.i, self.j = i, j
        x0, y0 = j * pieceLen, i * pieceLen
        x1, y1 = x0 + pieceLen, y0 + pieceLen
        pieceArray = imgArray[y0:y1, x0:x1]
        left, right = pieceArray[:, 0], pieceArray[:, -1]
        bottom, top = pieceArray[-1, :], pieceArray[0, :]
        bottom, left = np.flipud(bottom), np.flipud(left)
        #print('start')
        #print(bottom)
        # 0 for top, 1 for right, 2 for bottom, 3 for left

        self.cols = np.array([top, right, bottom, left])

        #print("break")

        leftInt, rightInt = pieceArray[:, 1], pieceArray[:, -2]
        bottomInt, topInt = pieceArray[-2, :], pieceArray[1, :]
        bottomInt, leftInt = np.flipud(bottomInt), np.flipud(leftInt)
        #print(bottomInt)
        #print("end")

        interiors = np.array([topInt, rightInt, bottomInt, leftInt])

        means, covInvs = [], []
        for x in range(4):
            gradArray = np.transpose(self.cols[x] - interiors[x])
            #print("x%d" % x)
            #print(self.cols[x])
            #print(interiors[x])
            #print("subtract")
            #print(self.cols[x] - interiors[x])
            #print()
            means.append(np.mean(gradArray, 1))
            covInvs.append(_getInverseCovariance(gradArray))

        #print("i: %d, j: %d, rgb: %r" % (i, j, self.cols))
        #print(np.nonzero(pieceArray == [116, 94, 79]))

        self.means = np.array(means)
        self.covInvs = np.array(covInvs)

    def __getitem__(self, key):
        return self.cols[key]


class _Edge:
    def __init__(self, piece1, piece2, orient1, orient2):
        self.pieces = (piece1, piece2)
        #self.orientation = orient1 * 4 + orient2
        self.orientation = (orient1, orient2)
        self.dissim = _Edge.symetricCompatMeasure(piece1, orient1,
                                                  piece2, orient2)

    def __lt__(self, other):  # to get heapq to work with this
        return self.dissim < other.dissim

    def __eq__(self, other):
        return self.dissim == other.dissim

    def sift(self, secondSmallest):
        self.dissim = self.dissim / secondSmallest.dissim

    @staticmethod
    def addEdgesToHeap(heap, piece1, piece2):
        edges = []
        for r1 in range(4):
            for r2 in range(4):
                newEdge = _Edge(piece1, piece2, r1, r2)
                #print("r1: %d, r2: %d, weight: %r" % (i, j, newEdge.dissim))
                edges.append(newEdge)
        edges = sorted(edges)
        min2 = edges[1]
        for edge in edges:
            edge.sift(min2)
            heapq.heappush(heap, edge)

    @staticmethod
    def getOrientations(orientation):
        orient1 = orientation / 4
        orient2 = orientation % 4
        return (orient1, orient2)

    @staticmethod
    def symetricCompatMeasure(leftPiece, leftOrient, rightPiece, rightOrient):

        grad_lr = np.flipud(leftPiece[leftOrient]) - rightPiece[rightOrient]
        grad_rl = np.flipud(rightPiece[rightOrient]) - leftPiece[leftOrient]

        d_lr= _compatMeasure(grad_lr, leftPiece.means[leftOrient],
                                leftPiece.covInvs[leftOrient])
        d_rl= _compatMeasure(grad_rl, rightPiece.means[rightOrient],
                                rightPiece.covInvs[rightOrient])
        #if a:
        #    print(grad_lr)
        #if b:
        #    print(grad_rl)
        return d_lr + d_rl


class JigsawTree:
    def __init__(self, inFilename, pieceLen):
        imIn = Image.open(inFilename)
        rows = int(imIn.size[1]/pieceLen)
        cols = int(imIn.size[0]/pieceLen)

        imIn.show()

        imArray = np.asarray(imIn, dtype=np.int16)
        #print("rows %s, cols %s" % (rows, cols))
        #print(imArray.shape)


        self.edges = []
        pieces = []
        for i in range(rows):
            for j in range(cols):
                newPiece = _Piece(imArray, i, j, pieceLen)
                for piece in pieces:
                    #print("i%d, j%d" % (i,j))
                    _Edge.addEdgesToHeap(self.edges, piece, newPiece)
                pieces.append(newPiece)


    def solve(self):
        while len(self.edges) > 0:
            edge = heapq.heappop(self.edges)
            p1, p2 = edge.pieces[0], edge.pieces[1]
            print("Edge weight: %r, piece1: (%d, %d), piece2: (%d, %d)"
                 % (edge.dissim, p1.i, p1.j, p2.i, p2.j))
            print("Rotation: %d, %d" % edge.orientation)

#TODO edges better, but still not matching up quite right seems like