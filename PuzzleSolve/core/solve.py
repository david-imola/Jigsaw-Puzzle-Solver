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

def translate_CCW_CW(rotations):
    if rotations == 2 or rotations == 0:
        return rotations
    elif rotations == 1:
        return 3
    elif rotations == 3:
        return 1

def array_conflict(array1, array2):
    return np.any(np.logical_and(array1, array2))


class _Piece:
    def __init__(self, imgArray, index, i, j, pieceLen, clusterCoord=(0, 0), orientation=0):
        self.index = index
        self.coord = (i, j)
        self.clusterCoord = clusterCoord
        self.orientation = orientation
        self.edges = [[], [], [], []] # top, left, bottom, righty
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
    def __init__(self, piece1, piece2, orient1, orient2=None, dissim=None):
        self.pieces = (piece1, piece2)
        if orient2 is None:
            self.orientation = orient1
        else:
            self.orientation = orient1 * 4 + orient2
        #self.orientation = (orient1, orient2)
        if dissim is None:
            self.dissim = _Edge.symetricCompatMeasure(piece1, orient1,
                                                      piece2, orient2)
        else:
            self.dissim = dissim

        

    def __lt__(self, other):  # to get heapq to work with this
        return self.dissim < other.dissim

    def __eq__(self, other):
        return self.dissim == other.dissim

    def sift(self, secondSmallest):
        self.dissim = self.dissim / secondSmallest.dissim

    @classmethod
    def copy(cls, instance):
        pieces = copy.copy(instance.pieces)
        orientation = copy.deepcopy(instance.orientation)
        dissim = copy.deepcopy(instance.dissim)

        return cls(pieces[0], pieces[1], orientation, None, dissim)

    @staticmethod
    def addToPieceEdges(leftPiece, rightPiece):
        #edges = [[], [], [], []]
        for r1 in range(4):
            for r2 in range(4):
                #print("r1: %d, r2: %d, weight: %r" % (i, j, newEdge.dissim))
                newEdgeLeft = _Edge(leftPiece, rightPiece, r1, r2)
                leftPiece.edges[r1].append(newEdgeLeft)
                newEdgeRight = _Edge.copy(newEdgeLeft)
                rightPiece.edges[r2].append(newEdgeRight)
        #edges = sorted(edges)
        #min2 = edges[1]
        #TODO ^ for efficienty's sake, make this an algorithm that only finds
        #The second minimum instead of sorting everything just for it.
        #for edge in edges:
        #    if (piece1.index == 9 and piece2.index == 5) or (piece1.index == 5 and piece2.index == 9):
        #        print(edge.dissim)
        #    edge.sift(min2)
        #    heapq.heappush(heap, edge)

        #print()

    def getOrientation(self):
        return _Edge.getOrientations(self.orientation)

    @staticmethod
    def getOrientations(orientation):
        orient1 = orientation / 4
        orient2 = orientation % 4
        return (int(orient1), int(orient2))

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
    def _prepOneArray(leftShape, rightArray, leftDestCoord, rightCoord):
        rowsMaybe = leftDestCoord[0] + rightArray.shape[0] - rightCoord[0]
        colsMaybe = leftDestCoord[1] + rightArray.shape[1] - rightCoord[1] - 1

        rows = leftShape[0] if leftShape[0] > rowsMaybe else rowsMaybe
        cols = leftShape[1] if leftShape[1] > colsMaybe else colsMaybe

        j_right, j_dest = rightCoord[1], leftDestCoord[1]
        diff_j = j_dest - j_right
        
        if diff_j > 0:
            empty_left = np.zeros((rightArray.shape[0], diff_j))
            preppedArray = np.concatenate((empty_left, rightArray), axis=1)
        else:
            preppedArray = rightArray

        remainingCols = cols - preppedArray.shape[1]
        if remainingCols > 0:
            empty_right = np.zeros((rightArray.shape[0], remainingCols))
            preppedArray = np.concatenate((preppedArray, empty_right), axis=1)

        i_right, i_dest = rightCoord[0], leftDestCoord[0]
        diff_i = i_dest - i_right

        if diff_i > 0:
            empty_top = np.zeros((diff_i, preppedArray.shape[1]))
            preppedArray = np.concatenate((empty_top, preppedArray), axis=0)

        remainingRows = rows - preppedArray.shape[0]
        if remainingRows > 0:
            empty_bottom = np.zeros((remainingRows, preppedArray.shape[1]))
            preppedArray = np.concatenate((preppedArray, empty_bottom), axis=0)

        return preppedArray

    @staticmethod
    def _prepArrays(leftArray, rightArray, leftCoord, rightCoord):
        leftDestCoord = (leftCoord[0], leftCoord[1] + 1)
        newRightArray = _Cluster._prepOneArray(leftArray.shape, rightArray,
                                            leftDestCoord, rightCoord)

        #Now, we flip both arrays and left becomes right and vice versa
        leftArray180 = np.rot90(leftArray, 2)
        leftCoord180 = coord_flip180(leftCoord, leftArray.shape)

        rightCoord180 = coord_flip180(rightCoord, rightArray.shape)
        rightDestCoord180 = (rightCoord180[0], rightCoord180[1] + 1)
        newLeftArray180 = _Cluster._prepOneArray(newRightArray.shape,
                                                 leftArray180,
                                                 rightDestCoord180,
                                                 leftCoord180)
        return (np.rot90(newLeftArray180, 2), newRightArray)

    @staticmethod
    def potentiallyMerge(edge):
        if edge.pieces[0].cluster is edge.pieces[1].cluster:
            return False  # That edge can be disregarded

        #print(edge.pieces[0].coord, edge.pieces[1].coord)

        #first check for merge conflict
        #if merge conflict, return false
        #otherwise, merge the arrays and update the rotationos and coords of the pieces

        leftCluster = edge.pieces[0].cluster
        rightCluster = edge.pieces[1].cluster

        orientLeft, orientRight = edge.getOrientation()

        #left_subtrahend = edge.pieces[0].orientation if hasattr(edge.pieces[0], "orientation") else orientLeft
        #right_subtrahend = edge.pieces[1].orientation if hasattr(edge.pieces[1], "orientation") else orientRight

        rotsLeft = (7 - edge.pieces[0].orientation - orientLeft) % 4
        rotsRight = (9 - edge.pieces[1].orientation - orientRight) % 4

        #print("rotsLeft, rotsRight (%d, %d)" % (rotsLeft, rotsRight))

        #print("edgeOrientLeft: %d, edgeOrientRight: %d" % (orientLeft, orientRight))

        #print("orieLeftPiece (%d)" % edge.pieces[0].orientation)
        #print("orieRightPiece (%d)" % edge.pieces[1].orientation)

        leftCoord_oriented = coord_rotate(edge.pieces[0].clusterCoord,
                                          rotsLeft, leftCluster.pieceArray.shape)
        rightCoord_oriented = coord_rotate(edge.pieces[1].clusterCoord,
                                           rotsRight, rightCluster.pieceArray.shape)
        
        leftArray_oriented = np.rot90(leftCluster.pieceArray, rotsLeft)
        rightArray_oriented = np.rot90(rightCluster.pieceArray, rotsRight)

        leftArray_filled, rightArray_filled =\
            _Cluster._prepArrays(leftArray=leftArray_oriented,
                                            rightArray=rightArray_oriented,
                                            leftCoord=leftCoord_oriented,
                                            rightCoord=rightCoord_oriented)

        #print("left_filled:\n",leftArray_filled)
        #print("right_filled\n", rightArray_filled)

        # Check for merge conflict
        if array_conflict(leftArray_filled, rightArray_filled):
            return False


        # if either of the pieces doesn't have an orientation already,
        # initialize it to the edge's orientation for that piece
        #if not hasattr(edge.pieces[0], "orientation"):
        #    edge.pieces[0].orientation = 0
        #if not hasattr(edge.pieces[1], "orientation"):
        #    edge.pieces[1].orientation = 0

        #Rotate the arrays' pieces
        leftCluster._rotatePieces(rotsLeft)
        rightCluster._rotatePieces(rotsRight)

        # if no conflict, we can go ahead and merge the arrays
        merged = np.where(leftArray_filled == 0,
                          rightArray_filled, leftArray_filled)

        #print(edge.pieces[0].index)
        #print(leftCluster.pieceArray)

        #print(edge.dissim)

        #print(edge.pieces[1].index)
        #print(rightCluster.pieceArray)

        #print(merged)
        #print()
        #print()
        # Delete the right piece's cluster: no longer neccessary as it will be merged
        #del rightCluster

        #Assign the new merged array as left's cluster
        leftCluster.pieceArray = merged

        #print("merged\n", merged)
        
        #Make the newly merged cluster's pieces realise their correct indeces
        leftCluster._updatePieces()

        #edge.pieces[1].cluster = leftCluster

        #print(edge.pieces[1].cluster.pieceArray)
        
        #print("orieLeftPiece (%d)" % edge.pieces[0].orientation)
        #print("orieRightPiece (%d)" % edge.pieces[1].orientation)

        #print()

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
                    #TODO ^ for efficienty's sake, make this an algorithm that only finds
                    #The second minimum instead of sorting everything just for it.
                    for edge in allEdges:
                        edge.sift(min2)
                        heapq.heappush(self.edges, edge)




        self.pieceCount = len(self.pieces) - 1  # exclude tne nullPiece
        for x in range(1, self.pieceCount + 1):
            _Cluster(self.pieces[x], self.rotatePiece, self.updatePiece)

        

    def showEdges(self):
        while len(self.edges) > 1000:
            edge = heapq.heappop(self.edges)
            p1, p2 = edge.pieces[0], edge.pieces[1]
            print("Edge weight: %r, piece1: (%d, %d), piece2: (%d, %d)"
                  % (edge.dissim, p1.coord[0], p1.coord[1], p2.coord[0], p2.coord[1]))
            print("Rotation: %d, %d" % edge.getOrientation())
        #TODO finsih up here

    def solve(self):
        #Will be 8*n*(n-1) edges produced
        clusterCount = self.pieceCount
        while clusterCount > 1:
            edge = heapq.heappop(self.edges)
            if _Cluster.potentiallyMerge(edge):
                clusterCount -= 1

        #for piece in self.pieces:
        #    if piece is not self.nullPiece:
        #        print(piece.index)
        #        print(piece.cluster.pieceArray)

    def showPuzzle(self):
        pieceArray = self.pieces[1].cluster.pieceArray
        shape = pieceArray.shape
        size = (shape[1] * self.pieceLen, shape[0] * self.pieceLen)
        solvedIm = Image.new("RGB", size)

        for (i,j), index in np.ndenumerate(pieceArray):
            if int(index) == 0:
                continue
            inX, inY = int((index - 1) % self.cols) * self.pieceLen, int((index - 1) / self.cols) * self.pieceLen
            data = (inX, inY, inX + self.pieceLen, inY + self.pieceLen)
            imPiece = self.imIn.transform((self.pieceLen, self.pieceLen), Image.EXTENT, data)
            numRots = self.pieces[int(index)].orientation
            #print(numRots)
            #print(translate_CCW_CW(numRots))
            #print()
            if numRots:
                imPiece = imPiece.rotate(90  * numRots)
            solvedCoord = (j * self.pieceLen, i * self.pieceLen)
            solvedIm.paste(imPiece, solvedCoord)

        #print(pieceArray)
        solvedIm.show("Solved Puzzle")



    def rotatePiece(self, index, rotation):
        newOrient = (self.pieces[index].orientation + rotation) % 4
        self.pieces[index].orientation = newOrient

    def updatePiece(self, cluster, index, coord):
        self.pieces[index].cluster = cluster
        self.pieces[index].clusterCoord = coord
