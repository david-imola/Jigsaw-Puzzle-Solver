from PIL import Image
import numpy as np





def _addRandomNoise(array):
    np.random.seed(0) # We actually want consistent noise
    noisey = np.random.randint(0, 2, size=(3, 9))
    return np.concatenate(array, noisey)

def _compatMeasure(gradArray):
    noiseyGrad = _addRandomNoise(gradArray)
    mean = np.mean(noiseyArray, axis=0)

    covInv = np.linalg(np.cov(noiseyGrad))

    score = 0
    P = noiseyArray.shape[0]
    for p in range(P):
        deviation = np.subtract(np.transpose(noiseyGrad), mean)
        term = np.matmul(deviation, covInv)
        score += np.matmul(term, deviation)

    return score




class _Piece:
    def __init__(self, imgArray, i, j, pieceLen):
        self.i, self.j = i, j
        x0, y0 = j * pieceLen, i * pieceLen
        x1, y1 = x0 + pieceLen, y0 + pieceLen
        pieceArray = imgArray[x0:x1, y0:y1]
        self.grad = []  # 0 for top, 1 for right, 2 for bottom, 3 for left
        self.grad[0] = np.subtract(pieceArray[:, 1], pieceArray[:, 0])
        self.grad[1] = np.subtract(pieceArray[:, pieceLen - 2],
                                  pieceArray[:, pieceLen - 1])
        self.grad[2] = np.subtract(pieceArray[pieceArray - 2, :],
                                   pieceArray[pieceArray - 1, :])
        self.grad[3] = np.subtract(pieceArray[1, :], pieceArray[0, :])

        for i in range(4):
            self.grad[i] = np.reshape(self.grad[i], newshape=(3, pieceLen))


class _Edge:
    def __init__(self, piece0, piece1):
        self.pieces = []
        self.piece[0] = piece0
        self.piece[1] = piece1
        # TODO implement cost function here, so that it can determine the orientation

    def __lt__(self, other): #to get heapq to work
        pass
    def __eq__(self, other):
        pass
    

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

