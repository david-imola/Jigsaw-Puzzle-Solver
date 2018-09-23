from PIL import Image
import numpy as np
from enum import Enum
import heapq
import copy


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


def coord_rotateCCW(coord, arrayShape):
    i_new = arrayShape[1] - coord[1] - 1
    j_new = coord[0]
    return (i_new, j_new)


def coord_rotateCW(coord, arrayShape):
    i_new = coord[1]
    j_new = arrayShape[0] - coord[0] - 1
    return (i_new, j_new)


def coord_flip180(coord, arrayShape):
    i_new = arrayShape[0] - coord[0] - 1
    j_new = arrayShape[1] - coord[1] - 1
    return (i_new, j_new)


def coord_rotate(coord, rotations, arrayShape):
    if rotations == 0:
        return coord
    elif rotations == 1:
        return coord_rotateCCW(coord, arrayShape)
    elif rotations == 2:
        return coord_flip180(coord, arrayShape)
    elif rotations == 3:
        return coord_rotateCW(coord, arrayShape)
    else:
        return False


def array_conflict(array1, array2):
    return np.any(np.logical_and(array1, array2))


class _Piece:
    def __init__(self, imgArray, index, i, j, pieceLen,
                 clusterCoord=(0, 0), orientation=0):
        self.index = index  # Index in our piece array
        self.clusterCoord = clusterCoord  # Coordinate in the piece's cluster
        self.orientation = orientation
        # ^ How many times this piece is rotated counter-cw within its cluster

        # "Edges" here refer to edges in a graph.
        # Each subarray here refers to the edges of this piece's
        # top, left, bottom and right sides respectively
        self.edges = [[], [], [], []]

        # Select the individual pieceArray from the whole image array
        x0, y0 = j * pieceLen, i * pieceLen
        x1, y1 = x0 + pieceLen, y0 + pieceLen
        pieceArray = imgArray[y0:y1, x0:x1]

        # Save the outermost rows/columns of the image to compute edges
        # 0 for top, 1 for left, 2 for bottom, 3 for right
        left, right = pieceArray[:, 0], pieceArray[:, -1]
        bottom, top = pieceArray[-1, :], pieceArray[0, :]
        # When the bottom and left sides are rotated into being the right side,
        # they get flipped upside-down. Hence we need to do flip them here.
        bottom, left = np.flipud(bottom), np.flipud(left)
        self.cols = np.array([top, left, bottom, right])

        # Find the interior rows/columns to compute the gradient
        leftInt, rightInt = pieceArray[:, 1], pieceArray[:, -2]
        bottomInt, topInt = pieceArray[-2, :], pieceArray[1, :]
        bottomInt, leftInt = np.flipud(bottomInt), np.flipud(leftInt)
        interiors = np.array([topInt, leftInt, bottomInt, rightInt])

        means, covInvs = [], []  # The mean and inverse covariance of each side
        for x in range(4):
            gradArray = np.transpose(self.cols[x] - interiors[x])
            means.append(np.mean(gradArray, 1))
            covInvs.append(_getInverseCovariance(gradArray))

        self.means = np.array(means)
        self.covInvs = np.array(covInvs)

    def __getitem__(self, key):
        return self.cols[key]


class _Edge:
    def __init__(self, piece1, piece2, orient1, orient2=None, dissim=None):
        self.pieces = (piece1, piece2)
        if orient2 is None:
            self.orientation = orient1
        else:
            # To save a little memory, store the orientations in one base-4 int
            self.orientation = orient1 * 4 + orient2
        if dissim is None:
            self.dissim = _Edge.symetricCompatMeasure(piece1, orient1,
                                                      piece2, orient2)
        else:
            self.dissim = dissim

    def __lt__(self, other):
        return self.dissim < other.dissim

    def __eq__(self, other):
        return self.dissim == other.dissim

    def sift(self, secondSmallest):
        self.dissim = self.dissim / secondSmallest.dissim

    def getOrientation(self):
        '''
        Returns a tuple of the orientation for the left and right pieces.
        0 for top, 1 for left, 2 for  bottom, 3 for right.
        So if this returned the tuple (2, 3), then this edge's left piece's
        bottom side is being considered against the right piece's right side.
        '''
        # Convert our single 2-digit base-4 back into two distinct numbers
        orient1 = self.orientation / 4
        orient2 = self.orientation % 4
        return (int(orient1), int(orient2))

    @classmethod
    def copy(cls, instance):
        """
        Creates a copy of an edge. Keeps the references to the pieces the same, but
        creates a unique copy of the orientations and weight (dissimulatiry measure).
        Use to save time when you don't need to recalculate an edge weight, but want to
        have two copies of the edge, one for each piece. This way, when you sift, you don't
        end up altering the same edge twice (one time for each piece).
        """
        pieces = copy.copy(instance.pieces)
        orientation = copy.deepcopy(instance.orientation)
        dissim = copy.deepcopy(instance.dissim)

        return cls(pieces[0], pieces[1], orientation, None, dissim)

    @staticmethod
    def addToPieceEdges(leftPiece, rightPiece):
        """
        Builds edges from all combinations of the two pieces,
        then stores the edges in the two pieces
        """
        for r1 in range(4):
            for r2 in range(4):
                newEdgeLeft = _Edge(leftPiece, rightPiece, r1, r2)
                leftPiece.edges[r1].append(newEdgeLeft)

                # Here we copy, instead of initializing again a la above,
                # So we don't have to compute the wieght again.
                newEdgeRight = _Edge.copy(newEdgeLeft)
                rightPiece.edges[r2].append(newEdgeRight)


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
    def __init__(self, initPiece, pieceRotateCallback, pieceUpdateCallback):
        self.pieceArray = np.array([initPiece.index], ndmin=2)
        initPiece.cluster = self

        self.pieceRotateClbk = pieceRotateCallback
        self.pieceUpdateClbk = pieceUpdateCallback

    def _rotatePieces(self, rotations):
        for _, index in np.ndenumerate(self.pieceArray):
            if index != 0:
                self.pieceRotateClbk(int(index), rotations)

    def _updatePieces(self):
        for coord, index in np.ndenumerate(self.pieceArray):
            if index != 0:
                self.pieceUpdateClbk(self, int(index), coord)

    @staticmethod
    def _prepOneArray(leftShape, rightArray, leftCoord, rightCoord):
        # i_0 + height_0 - i_1
        # If the left array only had the column of the left piece,
        # this would be how many total rows would be required
        rowsMaybe = leftCoord[0] + rightArray.shape[0] - rightCoord[0]
        # ...But because there's more than one column in the left array
        # One of those columns might make the array taller
        rows = leftShape[0] if leftShape[0] > rowsMaybe else rowsMaybe

        colsMaybe = leftCoord[1] + rightArray.shape[1] - rightCoord[1] + 1
        cols = leftShape[1] if leftShape[1] > colsMaybe else colsMaybe

        # How many zero columns to the left of the mass
        # Add the one here because its being placed in the next column over
        diff_j = leftCoord[1] - rightCoord[1] + 1
        if diff_j > 0:
            empty_left = np.zeros((rightArray.shape[0], diff_j))
            preppedArray = np.concatenate((empty_left, rightArray), axis=1)
        else: # this could happen when width_left < width_right
            preppedArray = rightArray

        # How many columns of zero to the right the mass
        remainingCols = cols - preppedArray.shape[1]
        if remainingCols > 0:
            empty_right = np.zeros((rightArray.shape[0], remainingCols))
            preppedArray = np.concatenate((preppedArray, empty_right), axis=1)

        # How many rows of zero above the mass
        diff_i = leftCoord[0] - rightCoord[0]
        if diff_i > 0:  # this would be false if height_left < height_right
            empty_top = np.zeros((diff_i, preppedArray.shape[1]))
            preppedArray = np.concatenate((empty_top, preppedArray), axis=0)

        # How many rows of zero below the mass
        remainingRows = rows - preppedArray.shape[0]
        if remainingRows > 0:
            empty_bottom = np.zeros((remainingRows, preppedArray.shape[1]))
            preppedArray = np.concatenate((preppedArray, empty_bottom), axis=0)

        return preppedArray

    @staticmethod
    def _prepArrays(leftArray, rightArray, leftCoord, rightCoord):
        newRightArray = _Cluster._prepOneArray(leftArray.shape, rightArray,
                                               leftCoord, rightCoord)

        # Now, we flip both arrays and left becomes right and vice versa
        leftArray180 = np.rot90(leftArray, 2)
        leftCoord180 = coord_flip180(leftCoord, leftArray.shape)

        rightCoord180 = coord_flip180(rightCoord, rightArray.shape)
        newLeftArray180 = _Cluster._prepOneArray(newRightArray.shape,
                                                 leftArray180,
                                                 rightCoord180,
                                                 leftCoord180)
        return (np.rot90(newLeftArray180, 2), newRightArray)

    @staticmethod
    def potentiallyMerge(edge):
        if edge.pieces[0].cluster is edge.pieces[1].cluster:
            # This edge to be disregarded if already in the same cluster
            return False

        leftCluster = edge.pieces[0].cluster
        rightCluster = edge.pieces[1].cluster

        orientLeft, orientRight = edge.getOrientation()

        # How many rotations are required to flip this edge's
        # left piece's side into being the right side of that piece
        rotsLeft = (7 - edge.pieces[0].orientation - orientLeft) % 4

        # How many rotations required to rotate the right piece's side
        # in question into being its left side (counter-clockwise)
        rotsRight = (9 - edge.pieces[1].orientation - orientRight) % 4

        # The cluster coordinate of the left piece in question
        # after being rotated
        leftCoord_oriented = coord_rotate(edge.pieces[0].clusterCoord,
                                          rotsLeft,
                                          leftCluster.pieceArray.shape)
        # The coord of the right piece after being rotated to its left side
        # within it's cluster, in terms of (row, column).
        rightCoord_oriented = coord_rotate(edge.pieces[1].clusterCoord,
                                           rotsRight,
                                           rightCluster.pieceArray.shape)

        # The left and right cluster arrays after being rotated
        # so that the left cluster's piece's right side matches up with
        # the right cluster's piece's left side
        leftArray_oriented = np.rot90(leftCluster.pieceArray, rotsLeft)
        rightArray_oriented = np.rot90(rightCluster.pieceArray, rotsRight)

        # The arrays, padded with zeros, so that they're both the same size
        # And they lie within the same coordinate plane based on which pieces are to be matched.
        # Example: Given the arrays...
        # [[1, 2]   and   [[6, 7]
        #  [3, 4]]         [0, 9]]
        # If we were to match piece 4 with piece 6, the filled arrays would become...
        # [[1, 2, 0 ,0]   and  [[0, 0, 0, 0]
        #  [3, 4, 0, 0]         [0, 0, 6, 7]
        #  [0, 0, 0, 0]]        [0, 0, 0, 9]]
        leftArray_filled, rightArray_filled =\
            _Cluster._prepArrays(leftArray=leftArray_oriented,
                                 rightArray=rightArray_oriented,
                                 leftCoord=leftCoord_oriented,
                                 rightCoord=rightCoord_oriented)

        # Check for merge conflict
        if array_conflict(leftArray_filled, rightArray_filled):
            return False

        # Rotate the arrays' pieces
        leftCluster._rotatePieces(rotsLeft)
        rightCluster._rotatePieces(rotsRight)

        # if no conflict, we can go ahead and merge the arrays
        merged = np.where(leftArray_filled == 0,
                          rightArray_filled, leftArray_filled)

        # Assign the new merged array as left's cluster
        leftCluster.pieceArray = merged

        # Make the newly merged cluster's pieces realise
        # their new cluster and correct indeces.
        leftCluster._updatePieces()

        # A merge occured, so...
        return True


class JigsawTree:
    def __init__(self, inFilename, pieceLen):
        self.imIn = Image.open(inFilename)
        self.rows = int(self.imIn.size[1]/pieceLen)
        self.cols = int(self.imIn.size[0]/pieceLen)

        imArray = np.asarray(self.imIn, dtype=np.int16)

        self.pieceLen = pieceLen

        self.nullPiece = None
        self.pieces = [self.nullPiece]
        for i in range(self.rows):
            for j in range(self.cols):
                newPiece = _Piece(imArray, len(self.pieces), i, j, pieceLen)
                for piece in self.pieces:
                    if piece is not self.nullPiece:
                        _Edge.addToPieceEdges(piece, newPiece)
                self.pieces.append(newPiece)

        self.edges = []

        for piece in self.pieces:
            if piece is not self.nullPiece:
                for allEdges in piece.edges:
                    allEdges = sorted(allEdges)
                    min2 = allEdges[1]
                    # TODO ^ for efficienty's sake, make this an algorithm that only finds
                    # The second minimum instead of sorting everything just for it.
                    for edge in allEdges:
                        edge.sift(min2)
                        heapq.heappush(self.edges, edge)
        self.pieceCount = len(self.pieces) - 1  # exclude tne nullPiece
        for x in range(1, self.pieceCount + 1):
            _Cluster(self.pieces[x], self.rotatePiece, self.updatePiece)

    def solve(self):
        # Will be 8*n*(n-1) edges produced
        clusterCount = self.pieceCount
        while clusterCount > 1:
            edge = heapq.heappop(self.edges)
            if _Cluster.potentiallyMerge(edge):
                clusterCount -= 1

    def showPuzzle(self):
        pieceArray = self.pieces[1].cluster.pieceArray
        shape = pieceArray.shape
        size = (shape[1] * self.pieceLen, shape[0] * self.pieceLen)
        solvedIm = Image.new("RGB", size)

        for (i, j), index in np.ndenumerate(pieceArray):
            if int(index) == 0:
                continue
            inX = int((index - 1) % self.cols) * self.pieceLen
            inY = int((index - 1) / self.cols) * self.pieceLen
            data = (inX, inY, inX + self.pieceLen, inY + self.pieceLen)
            imPiece = self.imIn.transform((self.pieceLen, self.pieceLen),
                                          Image.EXTENT, data)
            numRots = self.pieces[int(index)].orientation
            if numRots:
                imPiece = imPiece.rotate(90 * numRots)
            solvedCoord = (j * self.pieceLen, i * self.pieceLen)
            solvedIm.paste(imPiece, solvedCoord)

        solvedIm.show("Solved Puzzle")

    def rotatePiece(self, index, rotation):
        newOrient = (self.pieces[index].orientation + rotation) % 4
        self.pieces[index].orientation = newOrient

    def updatePiece(self, cluster, index, coord):
        self.pieces[index].cluster = cluster
        self.pieces[index].clusterCoord = coord
