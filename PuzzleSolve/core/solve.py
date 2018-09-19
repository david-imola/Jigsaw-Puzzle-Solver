from PIL import Image
import numpy as np
from enum import Enum
import heapq


def _rotateArray(array, numRots):
    if numRots == 0:
        return array
    elif numRots == 1:
        pass

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
    for p in range(P):
        deviation = grad[p, :] - mean
        term = np.matmul(deviation, covInv)
        score += np.matmul(term, deviation)
    return score


class _Piece:
    def __init__(self, imgArray, index, i, j, pieceLen):
        self.index = index
        x0, y0 = j * pieceLen, i * pieceLen
        x1, y1 = x0 + pieceLen, y0 + pieceLen
        pieceArray = imgArray[y0:y1, x0:x1]
        left, right = pieceArray[:, 0], pieceArray[:, -1]
        bottom, top = pieceArray[-1, :], pieceArray[0, :]

        bottom, left = np.flipud(bottom), np.flipud(left)
        # 0 for top, 1 for left, 2 for bottom, 3 for bottom
        #(Counter-clockwise)
        self.cols = np.array([top, left, bottom, right])

        leftInt, rightInt = pieceArray[:, 1], pieceArray[:, -2]
        bottomInt, topInt = pieceArray[-2, :], pieceArray[1, :]
        bottomInt, leftInt = np.flipud(bottomInt), np.flipud(leftInt)

        interiors = np.array([topInt, leftInt, bottomInt, rightInt])

        means, covInvs = [], []
        for x in range(4):
            gradArray = np.transpose(self.cols[x] - interiors[x])
            means.append(np.mean(gradArray, 1))
            covInvs.append(_getInverseCovariance(gradArray))

        self.means = np.array(means)
        self.covInvs = np.array(covInvs)

    def __getitem__(self, key):
        return self.cols[key]

class _Edge:
    def __init__(self, piece1, piece2, orient1, orient2):
        self.pieces = (piece1, piece2)
        self.orientation = orient1 * 4 + orient2
        #self.orientation = (orient1, orient2)
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

    def getOrientation(self):
        return _Edge.getOrientations(self.orientation)

    @staticmethod
    def getOrientations(orientation):
        orient1 = orientation / 4
        orient2 = orientation % 4
        return (orient1, orient2)

    @staticmethod
    def symetricCompatMeasure(leftPiece, leftOrient, rightPiece, rightOrient):

        grad_lr = np.flipud(leftPiece[leftOrient]) - rightPiece[rightOrient]
        grad_rl = np.flipud(rightPiece[rightOrient]) - leftPiece[leftOrient]

        d_lr = _compatMeasure(grad_lr, leftPiece.means[leftOrient],
                              leftPiece.covInvs[leftOrient])
        d_rl = _compatMeasure(grad_rl, rightPiece.means[rightOrient],
                              rightPiece.covInvs[rightOrient])
        return d_lr + d_rl

class _Cluster:
    def __init__(self, initPiece, pieceIndex):
        self.pieces = np.array([pieceIndex, 0], ndmin=3)
        initPiece.cluster = self


    @staticmethod
    def _prepOneArray(leftShape, rightArray, leftDestCoord, rightCoord):
        rowsMaybe = leftDestCoord[0] + rightArray.shape[0] - rightCoord[0]
        colsMaybe = leftDestCoord[1] + rightArray.shape[1] - rightCoord[1] - 1

        rows = leftShape[0] if leftShape[0] > rowsMaybe else rowsMaybe
        cols = leftShape[1] if leftShape[1] > colsMaybe else colsMaybe

        #print(leftDestCoord)

        #print(leftDestCoord)
        #print(rightCoord)
        j_right, j_dest = rightCoord[1], leftDestCoord[1]
        diff_j = j_dest - j_right
        
        #print((rightArray.shape[0], diff_j))
        #TODO make sure there diff_j actually > 0
        if diff_j > 0:
            empty_left = np.zeros((rightArray.shape[0], diff_j))
            preppedArray = np.concatenate((empty_left, rightArray), axis=1)
        else:
            preppedArray = rightArray

        remainingCols = cols - preppedArray.shape[1]
        if remainingCols > 0:
            empty_right = np.zeros(shape=(rightArray.shape[0], remainingCols))
            preppedArray = np.concatenate((preppedArray, empty_right), axis=1)

        i_right, i_dest = rightCoord[0], leftDestCoord[0]
        diff_i = i_dest - i_right

        #print(rightArray)
        #print(leftDestCoord, rightCoord)

        if diff_i > 0:
            empty_top = np.zeros(shape=(diff_i, preppedArray.shape[1]))
            preppedArray = np.concatenate((empty_top, preppedArray), axis=0)


        remainingRows = rows - preppedArray.shape[0]
        if remainingRows > 0:
            empty_bottom = np.zeros(shape=(remainingRows, preppedArray.shape[1]))
            preppedArray = np.concatenate((preppedArray, empty_bottom), axis=0)

        #print(preppedArray)
        return preppedArray

    @staticmethod
    def _prepArrays(leftArray, rightArray, leftCoord, rightCoord):
        leftDestCoord = (leftCoord[0], leftCoord[1] + 1)
        newRightArray = _Cluster._prepOneArray(leftArray.shape, rightArray, leftDestCoord, rightCoord)

        #Now, we flip both arrays and left becomes right and vice versa
        leftArray180 = np.rot90(leftArray, 2)
        height_left, width_left = leftArray.shape[0], leftArray.shape[1]
        leftCoord180 = (height_left - leftCoord[0] -1, width_left - leftCoord[1] -1)

        height_right, width_right = rightArray.shape[0], rightArray.shape[1]
        rightDestCoord180 = (height_right - rightCoord[0] - 1, width_right - rightCoord[1])
        newLeftArray180 = _Cluster._prepOneArray(newRightArray.shape, leftArray180, rightDestCoord180, leftCoord180)

        return (np.rot90(newLeftArray180, 2), newRightArray)


    @staticmethod
    def potentiallyMerge(edge):
        if edge.pieces[0].cluster is edge.pieces[1].cluster:
            return False #That edge can be disregarded
        #Returns true if a merge occured, false otherwise
        rot1, rot2 = edge.getOrientation()
        rots = (6 + rot1 - rot2) % 4
        #pieceRot = pieceRot + rot % 4 
        # TODO finish piece/cluster rotation


class JigsawTree:
    def __init__(self, inFilename, pieceLen):
        imIn = Image.open(inFilename)
        rows = int(imIn.size[1]/pieceLen)
        cols = int(imIn.size[0]/pieceLen)

        imArray = np.asarray(imIn, dtype=np.int16)

        self.edges = []
        self.pieces = []
        for i in range(rows):
            for j in range(cols):
                newPiece = _Piece(imArray, len(self.pieces), i, j, pieceLen)
                for piece in self.pieces:
                    _Edge.addEdgesToHeap(self.edges, piece, newPiece)
                self.pieces.append(newPiece)
        self.pieceCount = len(self.pieces)
        for x in range(self.pieceCount):
            _Cluster(self.pieces[x], x)

    def solve(self):
        treeCount = self.pieceCount
        c = 0
        while c < 7:
            edge = heapq.heappop(self.edges)
            if _Cluster.potentiallyMerge(edge):
                treeCount -= 1
            p1, p2 = edge.pieces[0], edge.pieces[1]
            print("Edge weight: %r, piece1: (%d), piece2: (%d)"
                 % (edge.dissim, p1.index, p2.index))
            print("Rotation: %d, %d" % edge.getOrientation())
            c += 1
