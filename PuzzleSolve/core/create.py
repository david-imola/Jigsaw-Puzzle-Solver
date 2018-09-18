from PIL import Image
import numpy


def _GetRandomIndex(row, actvEls=None):
    # 'actvEls': The # of elements in the row who are nonzero
    if not actvEls:
        actvEls = numpy.count_nonzero(row)
    if actvEls <= 0:
        return -1
    limit = (numpy.random.randint(1, actvEls)) if actvEls > 1 else 1
    counter, index = 0, 0

    while True:
        if index >= len(row):  # if index has gone past len...
            index = 0  # start it from 0
        if row[index]: #if nonzero (active) element,
            counter += 1 # increment counter
        if counter >= limit: # if we've incremented enough to satisfy the random limit...
            break
        else:
            index += 1 # keep going
    return index


def _createPiece(imIn, imOut, inI, inJ, outI, outJ, pieceLen):
    inX, inY = inJ * pieceLen, inI * pieceLen
    data = (inX, inY, inX + pieceLen, inY + pieceLen)
    piece = imIn.transform((pieceLen, pieceLen), Image.EXTENT, data)
    numRots = numpy.random.randint(0, 4)
    if numRots:
        piece = piece.rotate(90 * numRots)
    outCoord = (outJ * pieceLen, outI * pieceLen)
    imOut.paste(piece, outCoord)


def CreatePuzzle(inFilename, outFilename, pieceLen):
    imIn = Image.open(inFilename)  # input image

    rows = int(imIn.size[1]/pieceLen)
    cols = int(imIn.size[0]/pieceLen)

    imOut = Image.new(imIn.mode, (cols * pieceLen, rows * pieceLen))  # output image

    # An array to store the states of each jigsaw square...
    array = numpy.ones(shape=(rows, cols))
    # ... ie: if array[y][x] is 1, then the jigsaw piece at (x, y)
    # still needs to be created.
    # Upon this piece being created, array[y][x] should flip to 0.
    # Thus the final value of 'array' should be an array of zeros.
    # This is all neccesarry because we want the jigsaw pieces
    # to be produced in a random order.
    
    # vector to store the # of active elements of each row...
    rowLens = [cols] * rows
    # ... ie: given 'array' of:
    # [0 1 1]
    # [1 0 0]
    # [0 0 0]
    # rowLens should be the vector (2, 1, 0)
    # which is the number of active (nonzero) elements per row.

    # the number of active rows in rowLens...
    activeRowCnt = rows
    # ... ie: given the 'array' from the comment block above,
    # activeRowCnt should be 2, which is the # of rows
    # who contain at least one nonzero element.
    outI = _GetRandomIndex(rowLens, activeRowCnt)
    for inI in range(rows):
        for inJ in range(cols):
            outJ = _GetRandomIndex(array[outI], rowLens[outI])
            _createPiece(imIn, imOut, inI, inJ, outI, outJ, pieceLen)
            array[outI][outJ] = 0  # Switch that piece to inactive
            rowLens[outI] -= 1  # Decrement the # of active pieces in that row
            if rowLens[outI] <= 0:  # If that whole row is now dead...
                activeRowCnt -= 1  # Decrement the # of active rows
            outI = _GetRandomIndex(rowLens, activeRowCnt)
            
    imOut.save(outFilename, "PNG") #Must be png for losless compression (or else the pieces bleed into each other)
