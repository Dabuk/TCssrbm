import numpy
from PIL import Image
import scipy
import theano
import itertools
import sys
import os
import gc
import networkx


def getOutOfBoundSubArray(array, position, size):

    """
    array    : original array from which to sample the sub-array
    position : tuple containing the coordinates of the top-left corner of the
               desired sub-array.
    size     : tuple containing the size of the desired sub-array.


    Returns the specified sub array from the original array.
    """

    result = numpy.zeros(size, numpy.uint8)
    resultMask = numpy.zeros(size, numpy.uint8)

    ones = numpy.ones(size[2:])
    

    for i in range(size[0]):
        for j in range(size[1]):

            row = position[0] + i
            col = position[1] + j

            if (row >= 0 and col >= 0 and row < array.shape[0] and
                col < array.shape[1]):
            
                result[i][j] = array[row][col]
                resultMask[i][j] = ones
                
    return (result, resultMask)


def getWrappingSubArray(array, position, size):

    """
    array    : original array from which to sample the sub-array
    position : tuple containing the coordinates of the top-left corner of the
               desired sub-array.
    size     : tuple containing the size of the desired sub-array.


    Returns the specified sub array from the original array.
    """

    result = numpy.zeros(size, numpy.uint8)
    resultMask = numpy.ones(size, numpy.uint8)

    for i in range(size[0]):
        for j in range(size[1]):
            result[i][j] = array[(position[0] + i) % array.shape[0]][(position[1] + j) % array.shape[1]]

    return (result, resultMask)

   
def generateWeightedNeighborhoodArray(size, full, inc=0, patchSize=1,
                                      mergeSize=0):

    # Generate the basic neighborhood layout with 1s and 0s.
    if full:
        result = numpy.ones(size, numpy.int32)
    else:
        result = numpy.zeros(size, numpy.int32)
        result[:size[0] / 2 - (patchSize - 1) / 2 + mergeSize] += 1
        result[size[0] / 2 - (patchSize - 1) / 2 + mergeSize:
               size[0] / 2 - (patchSize - 1) / 2 + patchSize,
               :size[1] / 2 - (patchSize - 1) / 2 + mergeSize] += 1

    # Increment the weights according to inc's value
    for i in range(min(size[0] / 2, size[1] / 2)):
        center = result[size[0] / 2 - i:size[0] / 2 + i + 1,
                        size[1] / 2 - i:size[1] / 2 + i + 1]

        centerFilter = center != 0
        fullFilter = result != 0

        result[size[0] / 2 - i:size[0] / 2 + i + 1,
               size[1] / 2 - i:size[1] / 2 + i + 1] += inc * centerFilter

        if inc < 0:
            result += -1 * inc * fullFilter

    return result
    

# Defines and returns a theano function for calculating the neighborhood match
# between a patch of noise and an array of sample patches
def neighborhoodMatching(samplesValue, neighborhoodValue):

    samples = theano.tensor.constant(samplesValue, 'samples')
    noise = theano.tensor.tensor3('noise', 'int32')

    noiseNeighborhood = theano.tensor.tensor3('noiseNeigh', 'int32')

    neighborhood = theano.tensor.constant(neighborhoodValue, 'neighborhood')

    mismatch = (samples - noise) * noiseNeighborhood * neighborhood

    output = theano.tensor.sum(mismatch ** 2, [2, 3, 4])
    best = theano.tensor.argmin(output)

    f = theano.function([noise, noiseNeighborhood], best)
    return f
    
    
# Defines and returns a theano function for calculating the neighborhood match
# between a patch of noise and an array of sample patches
def neighborhoodMatching2(neighborhoodValue):

    tensor5 = theano.tensor.TensorType('uint8', (False,)*5)

    samples = tensor5('samples')
    noise = theano.tensor.tensor3('noise', 'int16')

    noiseNeighborhood = theano.tensor.tensor3('noiseNeigh', 'int32')

    neighborhood = theano.tensor.constant(neighborhoodValue, 'neighborhood')

    mismatch = (samples - noise) * noiseNeighborhood * neighborhood

    output = theano.tensor.sum(mismatch ** 2, [2, 3, 4])
    best = theano.tensor.argmin(output)

    f = theano.function([samples, noise, noiseNeighborhood], best)
    return f 
    

def neighborhoodMatchingStructure(neighborhoodValue, structWeight):

    tensor5 = theano.tensor.TensorType('uint8', (False,)*5)

    samples = tensor5('samples')
    noise = theano.tensor.tensor3('noise', 'int16')
    noiseNeighborhood = theano.tensor.tensor3('noiseNeigh', 'int32')
    
    struct = theano.tensor.tensor3('structure', 'int32')
    structNeighborhood = theano.tensor.tensor3('structureNeigh', 'int32')
    

    neighborhood = theano.tensor.constant(neighborhoodValue, 'neighborhood')

    mismatch = (samples - noise) * noiseNeighborhood * neighborhood
    structMismatch = (samples - struct) * structNeighborhood
    
    totalMismatch = mismatch ** 2 + structWeight * structMismatch ** 2

    output = theano.tensor.sum(totalMismatch, [2, 3, 4])
    best = theano.tensor.argmin(output)

    f = theano.function([samples, noise, noiseNeighborhood, struct,
                         structNeighborhood], best)
    return f 
    

def synthesisTexture(swatch, structure, noise, neighborhood,
                     sequential, wrap, patchSize, mergeSize, mergeMode,
                     structWeight):

    """

    neighborhood = numpy array containing a mask indicating which pixels of the neirborhood,
                   relative to the position of the analyzed pixel, must be considered. In the
                   array, the '1's indicate that the corresponding pixel must be considered,
                   the '0's indicate that the corresponding pixels can be ignored. Ex:

                   [[1 1 1 1 1]
                    [1 1 1 1 1]
                    [1 1 0 0 0]
                    [0 0 0 0 0]
                    [0 0 0 0 0]]

    sequential =   boolean. If true, the pixels are updated one after the other in order (from
                   top to bottom, from left to right). If false, the pixels are all updated at
                   the same time.
    """

    arraySamplingOp = None
    if wrap:
        arraySamplingOp = getWrappingSubArray
    else:
        arraySamplingOp = getOutOfBoundSubArray

    newPixelValues = numpy.zeros(noise.shape, numpy.int32)
    initMap = numpy.zeros(noise.shape[:2], numpy.int32)
    sourceMap = numpy.zeros(noise.shape[:2] + (3,), numpy.int32)
    seamMap = numpy.zeros((noise.shape[0] * 2, noise.shape[1] * 2), numpy.int32)
    seamMapInfo = numpy.asarray([[None for i in range(noise.shape[1] * 2)]
                                 for j in range(noise.shape[0] * 2)])

    noiseCopy = numpy.copy(noise)

    # Presample every possible subarray of the same size as neighborhood from the swatch.
    # It will save a LOT of time later              
    samples = numpy.asarray([[swatch[i:i+neighborhood.shape[0],j:j+neighborhood.shape[1]]
                              for i in range(len(swatch) - neighborhood.shape[0] + 1)]
                             for j in range(len(swatch[0]) - neighborhood.shape[1] + 1)])
    
    # Compile the theano function for neighborhood matching
    #neighMatchFn = neighborhoodMatching(samples, neighborhood)
    neighMatchFn = neighborhoodMatching2(neighborhood)
    neighMatchStructFn = neighborhoodMatchingStructure(neighborhood, structWeight)

    # Compute utility values for patch matching
    nbRowNoise = len(noiseCopy)
    nbColNoise = len(noiseCopy[0])
    halfPatchSize = (patchSize - 1) / 2

    # For every patch in the noise image
    rowNoise = 0
    while rowNoise < nbRowNoise:

        print rowNoise * nbRowNoise, '/', nbRowNoise * nbColNoise

        colNoise = 0
        while colNoise < nbColNoise:

            # Obtain the neighborhood of the next patch, in the noise image,
            # for which to compute the new pixel values.
            noiseNeigh, noiseMask = arraySamplingOp(noiseCopy,
                                                    (rowNoise - neighborhood.shape[0] / 2 + halfPatchSize,
                                                     colNoise - neighborhood.shape[1] / 2 + halfPatchSize),
                                                    neighborhood.shape)
                                                    
            # Obtain the neighborhood of the next patch, in the noise image,
            # for which to compute the new pixel values.
            structPatch, structMask = arraySamplingOp(structure,
                                                      (rowNoise - neighborhood.shape[0] / 2 + halfPatchSize,
                                                       colNoise - neighborhood.shape[1] / 2 + halfPatchSize),
                                                      neighborhood.shape)

            # Find the nearest neighbor in the swatch           
            #bestNeigh = neighMatchFn(noiseNeigh, noiseMask)
            #bestNeigh = neighMatchFn(samples, noiseNeigh, noiseMask)
            bestNeigh = neighMatchStructFn(samples, noiseNeigh, noiseMask, structPatch, structMask)

            bestSample = samples[bestNeigh / samples.shape[1],
                                 bestNeigh % samples.shape[1]]

            # Extract, from the best neighborhood, the best pixel values for the current patch.
            bestPatch = bestSample[neighborhood.shape[0]/2 - halfPatchSize:
                                   neighborhood.shape[0]/2 + patchSize/2 + 1,
                                   neighborhood.shape[1]/2 - halfPatchSize:
                                   neighborhood.shape[1]/2 + patchSize/2 + 1]

            bestPatch = bestPatch[:nbRowNoise - rowNoise,
                                  :nbColNoise - colNoise]

            # Merge the patch with the part of the texture that has already
            # been generated
            if mergeMode == "overwrite":
                # Copy the new pixel values in the current patch of the
                # texture being generated
                newPixelValues[rowNoise:rowNoise + bestPatch.shape[0],
                               colNoise:colNoise + bestPatch.shape[1]] = bestPatch

                # If sequential, also copy the pixel values in the noise array
                if sequential:

                    noiseCopy[rowNoise:rowNoise + bestPatch.shape[0],
                               colNoise:colNoise + bestPatch.shape[1]] = bestPatch

            elif mergeMode == "graphcut":
                mergePatch(newPixelValues, seamMap, seamMapInfo, initMap,
                           sourceMap, bestPatch, rowNoise, colNoise)

                if sequential:
                    noiseCopy = numpy.copy(newPixelValues)

            # Increment colNoise
            colNoise += patchSize - mergeSize

        rowNoise += patchSize - mergeSize

    # Display the pre-"improvement phase" quality of the generated texture
    quality = (seamMap.sum(), (seamMap ** 2).sum())
    print "Pre-improvement : " + str(quality)

    # Iteratively improve the generated texture
    if mergeMode == "graphcut":

        regionSize = patchSize
        improvementNeigh = numpy.ones((regionSize, regionSize, 1), 
                                      dtype='int32')

        for noImprovement in range(00):

            # Find the worst region in the texture
            worstRegionCoord = (-1, -1)
            worstSeamTotal = -1
            
            for row in range(newPixelValues.shape[0] - regionSize):
                for col in range(newPixelValues.shape[1] - regionSize):
                                                   
                    total = seamMap[2 * row + 4: 2 * (row + regionSize) - 4,
                                    2 * col + 4: 2 * (col + regionSize) - 4].sum()

                    if total > worstSeamTotal:
                        worstRegionCoord = (row, col)
                        worstSeamTotal = total
                                    
            worstRegion = arraySamplingOp(newPixelValues,
                                          (worstRegionCoord[0] - neighborhood.shape[0] / 2 + halfPatchSize,
                                           worstRegionCoord[1] - neighborhood.shape[1] / 2 + halfPatchSize),
                                          neighborhood.shape)[0]
                                          
            # Find the best patch in the swatch
            bestNeigh = neighMatchFn(samples, worstRegion, improvementNeigh)

            bestSample = samples[bestNeigh / samples.shape[1],
                                 bestNeigh % samples.shape[1]]

            bestPatch = bestSample[neighborhood.shape[0] / 2 - halfPatchSize:
                                   neighborhood.shape[0] / 2 + patchSize / 2 + 1,
                                   neighborhood.shape[1] / 2 - halfPatchSize:
                                   neighborhood.shape[1] / 2 + patchSize / 2 + 1]

            # Merge the patch in the texture
            mergePatch(newPixelValues, seamMap, seamMapInfo, initMap,
                       sourceMap, bestPatch, worstRegionCoord[0], 
                       worstRegionCoord[1])

            # Display the new quality of the texture after the improvement
            quality = (seamMap.sum(), (seamMap ** 2).sum())
            print "Improvement #" + str(noImprovement) + " : " + str(quality)
            print "Region improved " + str(worstRegionCoord)

    return newPixelValues, [sourceMap, seamMap, seamMapInfo]


def mergePatch(texture, seamMap, seamMapInfo, initMap, sourceMap, patch,
               x, y):

    # Compute the overlap region
    overlapMask = initMap[x:x + patch.shape[0], y:y + patch.shape[1]]

    # If there is no merge to do, update the texture and initMap
    # and stop the function's execution
    if overlapMask.sum() == 0:
        texture[x: x + patch.shape[0], y: y + patch.shape[1]] = patch
        initMap[x: x + patch.shape[0], y: y + patch.shape[1]] = 1
        return 1

    # Generate the graph for max_flow min_cut algorithm
    g = networkx.Graph()

    for row in range(x, x + patch.shape[0]):
        for col in range(y, y + patch.shape[1]):

            # Check if the pixel is part of the overlap region
            if initMap[row, col] == 1:
                
                # Compute information on the pixel's neighbors
                topNeighInTexture = row > 0
                downNeighInTexture = row < texture.shape[0] - 1
                leftNeighInTexture = col > 0
                rightNeighInTexture = col < texture.shape[1] - 1
                        
                topNeighInPatch = row > x
                downNeighInPatch = row < x + patch.shape[0] - 1
                leftNeighInPatch = col > y
                rightNeighInPatch = col < y + patch.shape[1] - 1
                    
                topNeighInOverlap = (topNeighInTexture and
                                     topNeighInPatch and
                                     initMap[row - 1, col])
                downNeighInOverlap = (downNeighInTexture and 
                                      downNeighInPatch and
                                      initMap[row + 1, col])
                leftNeighInOverlap = (leftNeighInTexture and 
                                      leftNeighInPatch and
                                      initMap[row, col - 1])
                rightNeighInOverlap = (rightNeighInTexture and 
                                       rightNeighInPatch and
                                       initMap[row, col + 1])

                 # The texture is under construction
                if initMap.sum() < initMap.size:          

                    # Check if the pixel should be connected to the source
                    # node
                    connectToA = ((row == x and row != 0) or
                                  (col == y and col != 0))
                    
                    if connectToA:
                        add_edge(g, 'A', nameFromRowCol(row, col),
                                 capacity=999999.0)

                    # Check if the pixel should be connected to the target
                    # node
                    if (not connectToA and 
                        (not rightNeighInOverlap or not downNeighInOverlap)):
                        add_edge(g, 'B', nameFromRowCol(row, col),
                                 capacity=999999.0)

                # The texture has been fully constructed, it is under
                # improvement
                else:

                    # Check if the pixel should be connected to the source node
                    connectToA = (row == x or row == x + patch.shape[0] - 1 or
                                  col == y or col == y + patch.shape[1] - 1)
                    
                    if connectToA:
                        add_edge(g, 'A', nameFromRowCol(row, col),
                                 capacity=999999.0)
                                 
                # Connect the pixel to its right neighbor, if applicable
                if rightNeighInOverlap == 1:

                    # If there is already a seam between the pixel and it's
                    # right neighbor, add a special "seam node" to the graph.
                    # This node will be connected to the current pixel's node,
                    # its right neighbor's node as well as the 'B' node. Else,
                    # simply add an edge between the pixel and its neighbor.


                    if(seamMapInfo[row * 2, col * 2 + 1] != None):

                        seamValue = seamMap[row * 2, col * 2 + 1]
                        seamInfo = seamMapInfo[row * 2, col * 2 + 1]
                        seamNodeName = ("seam " + nameFromRowCol(row, col) + 
                                        " " + nameFromRowCol(row, col + 1))
                           
                        add_edge(g, seamNodeName, 'B', capacity=seamValue)
                        
                        add_edge(g,seamNodeName, nameFromRowCol(row, col),
                                   capacity=m(seamInfo[0], 
                                              patch[row - x, col - y],
                                              seamInfo[2],
                                              patch[row - x, col + 1 - y]))
                        
                        add_edge(g,seamNodeName, nameFromRowCol(row, col + 1),
                                   capacity=m(seamInfo[1], 
                                              patch[row - x, col - y],
                                              seamInfo[3],
                                              patch[row - x, col + 1 - y]))
                            
                    else:
                    
                        add_edge(g,nameFromRowCol(row, col),
                                   nameFromRowCol(row, col + 1),
                                   capacity=m(texture[row, col],
                                              patch[row - x, col - y],
                                              texture[row, col + 1],
                                              patch[row - x, col + 1 - y]))
                    
                    """
                    
                    add_edge(g,nameFromRowCol(row, col),
                               nameFromRowCol(row, col + 1),
                               capacity=m(texture[row, col],
                                          patch[row - x, col - y],
                                          texture[row, col + 1],
                                          patch[row - x, col + 1 - y]))
                    """
                                          

                # Connect the pixel to its bottom neighbor, if applicable
                if downNeighInOverlap == 1:
                    
                    # If there is already a seam between the pixel and it's
                    # bottom neighbor, add a special "seam node" to the graph.
                    # This node will be connected to the current pixel's node,
                    # its bottom neighbor's node as well as the 'B' node. 
                    # Else, simply add an edge between the pixel and its 
                    # neighbor.
                    
                    
                    if(seamMapInfo[row * 2 + 1, col * 2] != None):
                        
                        seamValue = seamMap[row * 2 + 1, col * 2]
                        seamInfo = seamMapInfo[row * 2 + 1, col * 2]
                        seamNodeName = ("seam " + nameFromRowCol(row, col) + 
                                        " " + nameFromRowCol(row + 1, col))
                           
                        add_edge(g, seamNodeName, 'B', capacity=seamValue)
                        
                        add_edge(g, seamNodeName, nameFromRowCol(row, col),
                                   capacity=m(seamInfo[0], 
                                              patch[row - x, col - y],
                                              seamInfo[2],
                                              patch[row + 1 - x, col - y]))
                        
                        add_edge(g, seamNodeName, nameFromRowCol(row + 1, col),
                                   capacity=m(seamInfo[1], 
                                              patch[row - x, col - y],
                                              seamInfo[3],
                                              patch[row + 1 - x, col - y]))
                            
                    else:
                    
                        add_edge(g, nameFromRowCol(row, col),
                                   nameFromRowCol(row + 1, col),
                                   capacity=m(texture[row, col],
                                              patch[row - x, col - y],
                                              texture[row + 1, col],
                                              patch[row + 1 - x, col - y]))
                    
                    """
                    
                    
                    add_edge(g,nameFromRowCol(row, col),
                               nameFromRowCol(row + 1, col),
                               capacity=m(texture[row, col],
                                          patch[row - x, col - y],
                                          texture[row + 1, col],
                                          patch[row + 1 - x, col - y]))
                    """

    if 'B' in g:

        # Compute max-flow min-cut
        try:
            flow = networkx.ford_fulkerson_flow(g, 'A', 'B')
        except:
            import pdb
            pdb.set_trace()

        # Compute the auxiliary graph and populate it with edges from the original
        # graph whose capacity are NOT saturated by the flow computed in the
        # max-flow algorithm.
        gAux = networkx.Graph()

        for edge in g.edges(data=True):                
            if (edge[2]['capacity'] - flow[edge[0]][edge[1]] >= 1e-10):
                gAux.add_edge(edge[0], edge[1])

        # Partition the auxiliary graph according to which nodes are connected to
        # the source node and which are connected to the target node.
        if 'A' in gAux:
            sourcePartition = set(networkx.single_source_shortest_path(gAux, 'A'))
        else:
            sourcePartition = set()
            
        if 'B' in gAux:
            targetPartition = set(networkx.single_source_shortest_path(gAux, 'B'))
        else:
            targetPartition = set() 
            
    else:
        sourcePartition = set(networkx.single_source_shortest_path(g, 'A'))
        targetPartition = set()

    # Ensure that the graph has been correctly separated in two
    # distinct partitions   
    assert(sourcePartition - targetPartition == sourcePartition)

    # Update the maps
    sourceColor = numpy.random.randint(0, 255, 3)
    oldPatch = numpy.copy(texture[x:x + patch.shape[0], y:y + patch.shape[1]])
    for row in range(x, x + patch.shape[0]):
        for col in range(y, y + patch.shape[1]):
            if initMap[row, col] == 0:
                sourceMap[row, col] = sourceColor
                texture[row, col] = patch[row - x][col - y]
            else:
                if nameFromRowCol(row, col) in targetPartition:
                    # This pixel receives it's color from the new patch
                    sourceMap[row, col] = sourceColor
                    texture[row, col] = patch[row - x][col - y]

                    # Update the seam between this pixel and its top neighbor
                    if nameFromRowCol(row - 1, col) in targetPartition:
                        seamMap[row * 2 - 1, col * 2] = 0.0
                        seamMapInfo[row * 2 - 1, col * 2] = None
                    else:

                        if row > x and initMap[row - 1, col]:
                            seamMap[row * 2 - 1, col * 2] = m(oldPatch[row - 1 - x, col - y],
                                                              patch[row - 1 - x,col - y],
                                                              oldPatch[row - x,col - y],
                                                              patch[row - x,col - y])

                            seamMapInfo[row * 2 - 1, col * 2] = (oldPatch[row - 1 - x,col - y],
                                                                 patch[row - 1 - x,col - y],
                                                                 oldPatch[row - x,col - y],
                                                                 patch[row - x,col - y])

                    # Update the seam between this pixel and its bottom
                    # neighbor
                    if nameFromRowCol(row + 1, col) in targetPartition:
                        seamMap[row * 2 + 1, col * 2] = 0.0
                        seamMapInfo[row * 2 + 1, col * 2] = None
                    else:

                        if row < x + patch.shape[0] - 1 and initMap[row+1, col]:
                            seamMap[row*2+1, col*2] = m(oldPatch[row-x,col-y],
                                                        patch[row-x,col-y],
                                                        oldPatch[row+1-x,col-y],
                                                        patch[row+1-x,col-y])

                            seamMapInfo[row*2+1, col*2] = (oldPatch[row-x,col-y],
                                                           patch[row-x,col-y],
                                                           oldPatch[row+1-x,col-y],
                                                           patch[row+1-x,col-y])

                    # Update the seam between this pixel and its left neighbor
                    if nameFromRowCol(row, col - 1) in targetPartition:
                        seamMap[row * 2, col * 2 - 1] = 0.0
                        seamMapInfo[row * 2, col * 2 - 1] = None
                    else:
                        if col > y and initMap[row, col - 1]:
                            seamMap[row*2, col*2-1] = m(oldPatch[row-x,col-1-y],
                                                        patch[row-x,col-1-y],
                                                        oldPatch[row-x,col-y],
                                                        patch[row-x,col-y])

                            seamMapInfo[row*2, col*2-1] = (oldPatch[row-x,col-1-y],
                                                           patch[row-x,col-1-y],
                                                           oldPatch[row-x,col-y],
                                                           patch[row-x,col-y])

                    # Update the seam between this pixel and its right neighbor
                    if nameFromRowCol(row, col + 1) in targetPartition:
                        seamMap[row * 2, col * 2 + 1] = 0.0
                        seamMapInfo[row * 2, col * 2 + 1] = None
                    else:
                        if (col < y + patch.shape[1] - 1 and
                            initMap[row, col + 1]):

                            seamMap[row*2, col*2+1] = m(patch[row-x,col-y],
                                                        oldPatch[row-x,col-y],
                                                        patch[row-x,col+1-y],
                                                        oldPatch[row-x,col+1-y])

                            seamMapInfo[row*2, col*2+1] = (patch[row-x,col-y],
                                                           oldPatch[row-x,col-y],
                                                           patch[row-x,col+1-y],
                                                           oldPatch[row-x,col+1-y])

            initMap[row, col] = 1

    return 1


def add_edge(graph, node1, node2, capacity):
    # This check is required because NetworkX raises exceptions when
    # an edge's capacity is equal or smaller than 0.
    if capacity > 0.0:
        graph.add_edge(node1, node2, capacity=capacity)


def m(As, Bs, At, Bt):
    """
    Pixel similarity function as defined in
    Graphcut Textures: Image and Video synthesis Using Graph Cuts
    """
    
    return numpy.linalg.norm(As - Bs) + numpy.linalg.norm(At - Bt)


def nameFromRowCol(row, col):
    return str(row) + '_' + str(col)


def rowColFromName(name):
    return (name[0], name[2])


def greenRedInterpolation(val, minVal, maxVal):
    rgb = numpy.zeros((3))

    # Constrain val between 0 and 1
    val = min(val, maxVal)
    val = max(val, minVal)
    normalizedVal = (val - minVal) / (maxVal - minVal)

    # Compute the corresponding color
    if normalizedVal < 0.5:
        rgb[0] = int(255 * 2 * normalizedVal)
        rgb[1] = 255
    else:
        rgb[0] = 255
        rgb[1] = 255 - int(255 * 2 * (normalizedVal - 0.5))

    return rgb

   
def loadImage(filename, size=None):
    
    img = Image.open(filename)
    
    if size != None:
        img = img.resize(size, Image.BICUBIC)
    
    imgArray = numpy.asarray(img)

    # If sample is grayscale, add a third dimension of size 1 to ensure
    # consistency between grayscale and color textures
    if len(imgArray.shape) == 2:
        imgArray = numpy.reshape(imgArray, imgArray.shape + (1,))
        
    return imgArray


def testTextureSynthesis(datasetName, textureName, textureSize, dictOrder,
                         dictWrap, dictNeighSize, dictNeighType, dictInput,
                         nbItt, dictPatchSize, dictNeighWeightInc,
                         dictMergeSize, dictMergeMode, dictStructWeight):

    # Generate a list of every possible combination of the parameters values
    # arrays
    paramCombinations = list(itertools.product(dictOrder, dictWrap,
                                                dictNeighSize,
                                                dictNeighType, dictInput,
                                                dictPatchSize,
                                                dictNeighWeightInc,
                                                dictMergeSize,
                                                dictMergeMode,
                                                dictStructWeight))

    """
    # Remove from the list the combinations that are invalid, superfluous
    # or give very poor results.
    paramCombinations = [p for p in paramCombinations if not (p[0] == 'parallel')]

    paramCombinations = [p for p in paramCombinations if not (p[1] == 'wrap')]

    paramCombinations = [p for p in paramCombinations if not (p[3] == 'full' and
                                                              p[4] == 'fromNoise')]

    paramCombinations = [p for p in paramCombinations if not (p[3] == 'partial' and
                                                              p[4] == 'fromSample')]
    """

    # Run texture synthesis for every valid combination of the parameters
    # values
    for params in paramCombinations:

        print "Starting test : " + "_".join(params)

        # Obtain the keys contained in params
        (orderKey, wrapKey, neighSizeKey, neighTypeKey,
         inputKey, patchSizeKey, neighIncKey, mergeSizeKey,
         mergeModeKey, structWeightKey) = params

        # Obtain the parameter values associated with the keys
        orderValue = dictOrder[orderKey]
        wrapValue = dictWrap[wrapKey]
        neighSizeValue = dictNeighSize[neighSizeKey]
        neighTypeValue = dictNeighType[neighTypeKey]
        inputValue = dictInput[inputKey]
        patchSizeValue = dictPatchSize[patchSizeKey]
        neighIncValue = dictNeighWeightInc[neighIncKey]
        mergeSizeValue = dictMergeSize[mergeSizeKey]
        mergeModeValue = dictMergeMode[mergeModeKey]
        structWeight = dictStructWeight[structWeightKey]

        # Load the texture sample from which to generate a new texture
        # and generate two swatches from it : one at ordinary resolution
        # and one at half the resolution.
        if datasetName == "brodatz":
            textureFilename = "/data/lisa/data/Brodatz/" + textureName + ".gif"
        elif datasetName == "ubi":
            textureFilename = "/data/lisa/data/ubi/textures/" + textureName + ".jpg"
        
        texture = loadImage(textureFilename)        
        textureNbChannels = texture.shape[2]
        swatch = texture[:min(texture.shape[0] / 2, textureSize),
                         :textureSize]
        
        
        # Load the low-res texture structure that the generated texture must
        # match        
        structFilename = "./LowResSample/" + datasetName + "/" + textureName + "/sample.png"
        structFilename = "./Samples_FullScale/" + datasetName + "/" + textureName + "/sample.png"
        structShape = loadImage(structFilename).shape
        struct = loadImage(structFilename, (structShape[0] * 2, structShape[1] * 2))
        struct = loadImage(structFilename)
        structNbChannels = struct.shape[2]
        
        

        # Load/generate the initial state for the new texture
        newTextureSize = 400
        newTextureInit = None
                
        if inputValue:
            # The initial state for the texture is random noise
            newTextureInit = numpy.random.randint(0, 256, (struct.shape))
            newTextureInit = numpy.asarray(newTextureInit, dtype='uint8')
        else:
            # The initial state for the texture is a sample previouly 
            # generated by a different model
            newTextureInit = loadImage("./Samples_FullScale/" + datasetName + 
                                       "/" + textureName + "/sample.png")
            newTextureInit = newTextureInit[0 : struct.shape[0], 
                                            0 : struct.shape[1]]
        newTextureNbChannels = newTextureInit.shape[2]


        # Ensure that the swatches and the new texture have the same number
        # of color channels
        assert(textureNbChannels == newTextureNbChannels)
        assert(textureNbChannels == structNbChannels)
        
        # Ensure that newTextureInit and struct have the same shape
        assert(newTextureInit.shape == struct.shape)

        # Generate the neighborhood       
        neighborhood = generateWeightedNeighborhoodArray(neighSizeValue +
                                                         newTextureInit.shape[2:],
                                                         neighTypeValue,
                                                         neighIncValue,
                                                         patchSizeValue,
                                                         mergeSizeValue)

        # Perform texture synthesis and save the result in a file whose name
        # contains the name of the parameter keys for easy identification
        newTexture = newTextureInit
        
        for i in range(1, nbItt + 1):
            newTexture, extraData = synthesisTexture(swatch,
                                                     struct,
                                                     newTexture,
                                                     neighborhood, orderValue,
                                                     wrapValue,
                                                     patchSizeValue,
                                                     mergeSizeValue,
                                                     mergeModeValue,
                                                     structWeight)

        sourceMap = extraData[0]
        seamMap = extraData[1]
        seamMapInfo = extraData[2]

        # If images are grayscale, remove the last dimensions to allow scipy
        # to save them properly
        if swatch.shape[2] == 1:
            swatch = swatch.mean(2)
            
        if newTexture.shape[2] == 1:
            newTexture = newTexture.mean(2)

        if newTextureInit.shape[2] == 1:
            newTextureInit = newTextureInit.mean(2)
            
        if struct.shape[2] == 1:
            struct = struct.mean(2)

        # Convert the seam map to color for higher visibility
        minSeam = 0
        white = numpy.asarray([255, 255, 255])
        black = numpy.asarray([0, 0, 0])
        maxSeam = m(white, black, white, black)
        colorSeamMap = numpy.zeros(seamMap.shape + (3,))
        for i in range(seamMap.shape[0]):
            for j in range(seamMap.shape[1]):
                if seamMapInfo[i][j] != None:
                    colorSeamMap[i, j] = greenRedInterpolation(seamMap[i, j],
                                                               minSeam,
                                                               maxSeam / 8)

        for i in range(newTexture.shape[0]):
            for j in range(newTexture.shape[1]):
                colorSeamMap[i * 2, j * 2] = newTexture[i, j]

        # Save the results and the inputs in a folder containing the name of
        # the texture and in files whose names contain the param dictionnary
        # keys for easy identification.
        folderName = "./results/" + textureName

        if not os.path.isdir(folderName):
            os.makedirs(folderName)

        newTextureFilename = folderName + "/output_" + '_'.join(params) + ".png"
        scipy.misc.imsave(newTextureFilename, newTexture)

        textureFilename = folderName + "/texture_" + '_'.join(params) + ".png"
        scipy.misc.imsave(textureFilename, swatch)

        originalFilename = folderName + "/noise_" + '_'.join(params) + ".png"
        scipy.misc.imsave(originalFilename, newTextureInit)

        sourceMapFilename = folderName + "/sourceMap_" + '_'.join(params) + ".png"
        scipy.misc.imsave(sourceMapFilename, sourceMap)
        
        structFilename = folderName + "/struct_" + '_'.join(params) + ".png"
        scipy.misc.imsave(structFilename, struct)

        seamMapFilename = folderName + "/seamMap_" + '_'.join(params) + ".png"
        scipy.misc.imsave(seamMapFilename, seamMap)

        seamMapColorFilename = folderName + "/seamMap_" + '_'.join(params) + "_color.png"
        scipy.misc.imsave(seamMapColorFilename, colorSeamMap)


def testTextureSynthesisMulti(datasetName, textureName, textureSize):

    # Instantiate parameter dictionaries
    dictOrder = {'parallel': False, 'sequential': True}
    dictWrap = {'wrap': True, 'nowrap': False}
    dictNeighSize = {'5x5': (5, 5), '9x9': (9, 9), '15x15': (15, 15)}
    dictNeighType = {'full': True, 'partial': False}
    dictInput = {'fromNoise': True, 'fromSample': False}
    nbItt = 1
    dictPatchSize = {'1x1patch': 1, '2x2patch': 2, '3x3patch': 3}
    dictNeighWeightInc = {"inc0": 0, "inc1": 1, "inc2": 2}
    dictMergeSize = {"mergeSize0": 0, "mergeSize8": 8}
    dictMergeMode = {"overwrite": "overwrite", "graphcut": "graphcut"}
    dictStructWeight = {"structW0": 0, "structW0": 0.125, "structW0": 0.25, 
                        "structW0": 0.5, "structW0": 1, "structW0": 2, 
                        "structW0": 4, "structW0": 8, "structW0": 16}
    


    dictOrder = {'sequential': True}
    dictWrap = {'nowrap': False}
    dictNeighSize = {'31x31': (31, 31)}
    dictNeighType = {'partial': False}
    dictInput = {'fromNoise': True}
    dictPatchSize = {'31x31patch': 31}
    dictMergeSize = {"mergeSize8": 8}
    dictMergeMode = {"graphcut": "graphcut"}
    dictNeighWeightInc = {"inc0": 0}
    dictStructWeight = {"structW0": 1}

    # Perform texture synthesis with the previously created parameter
    # dictionaries
    testTextureSynthesis(datasetName, textureName, textureSize, dictOrder,
                         dictWrap, dictNeighSize, dictNeighType, dictInput,
                         nbItt, dictPatchSize, dictNeighWeightInc,
                         dictMergeSize, dictMergeMode, dictStructWeight)


def testTextureSynthesisSingle(datasetName, textureName, textureSize,
                               sequentialOrder, wrap, neighSize, fullNeigh,
                               fromNoise, nbItt, patchSize, neighWeightInc,
                               mergeSize, mergeMode, dictStructWeight):

    # Instantiate parameter dictionaries

    if sequentialOrder == True:
        dictOrder = {'sequential': True}
    else:
        dictOrder = {'parallel': False}

    if wrap == True:
        dictWrap = {'wrap': True}
    else:
        dictWrap = {'nowrap': False}

    dictNeighSize = { str(neighSize) + 'x' + str(neighSize): (neighSize, neighSize)}

    if fullNeigh == True:
        dictNeighType = {'full': True}
    else:
        dictNeighType = {'partial': False}

    if fromNoise == True:
        dictInput = {'fromNoise': True}
    else:
        dictInput = {'fromSample': False}

    dictPatchSize = {str(patchSize) + 'x' + str(patchSize) + 'patch': patchSize}

    dictNeighWeightInc = {"inc" + str(neighWeightInc): neighWeightInc}

    dictMergeSize = {"mergeSize" + str(mergeSize): mergeSize}

    dictMergeMode = {mergeMode: mergeMode}
    
    dictStructWeight = {"structW" + str(structWeight): structWeight}

    # Perform texture synthesis with the previously created parameter
    # dictionaries
    testTextureSynthesis(datasetName, textureName, textureSize, dictOrder,
                         dictWrap, dictNeighSize, dictNeighType, dictInput,
                         nbItt, dictPatchSize, dictNeighWeightInc,
                         dictMergeSize, dictMergeMode, dictStructWeight)


def getParam(index, default):
    try:
        if sys.argv[index] == 'x':
            return default
        else:
            return sys.argv[index]
    except:
        return default


if __name__ == "__main__":

    numpy.random.seed(0)

    if sys.argv[1] == 'multi':

        # Extract input parameters
        datasetName = sys.argv[2]
        textureName = sys.argv[3]
        textureSize = int(sys.argv[4])

        # Perform texture synthesis with the given parameters
        testTextureSynthesisMulti(datasetName, textureName, textureSize)

    if sys.argv[1] == 'single':

        # Extract input parameters
        datasetName = sys.argv[2]
        textureName = sys.argv[3]
        textureSize = int(sys.argv[4])
        sequentialOrder = int(sys.argv[5]) != 0
        wrap = int(sys.argv[6]) != 0
        neighborhoodSize = int(sys.argv[7])
        fullNeighborhood = int(sys.argv[8]) != 0
        fromNoise = int(sys.argv[9]) != 0
        nbItt = int(sys.argv[10])
        patchSize = int(sys.argv[11])
        neighborhoodWeightInc = int(sys.argv[12])
        mergeSize = int(getParam(13, 0))
        mergeMode = getParam(14, "overwrite")
        structWeight = float(getParam(15, 1.0))

        # Perform texture synthesis with the given parameters
        testTextureSynthesisSingle(datasetName, textureName, textureSize,
                                   sequentialOrder, wrap, neighborhoodSize,
                                   fullNeighborhood, fromNoise, nbItt,
                                   patchSize, neighborhoodWeightInc,
                                   mergeSize, mergeMode, structWeight)
