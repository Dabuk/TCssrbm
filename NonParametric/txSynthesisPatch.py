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
    result_mask = numpy.zeros(size, numpy.uint8)

    ones = numpy.ones(size[2:])
    

    for i in range(size[0]):
        for j in range(size[1]):

            row = position[0] + i
            col = position[1] + j

            if (row >= 0 and col >= 0 and row < array.shape[0] and
                col < array.shape[1]):
            
                result[i][j] = array[row][col]
                result_mask[i][j] = ones
                
    return (result, result_mask)


def getWrappingSubArray(array, position, size):

    """
    array    : original array from which to sample the sub-array
    position : tuple containing the coordinates of the top-left corner of the
               desired sub-array.
    size     : tuple containing the size of the desired sub-array.


    Returns the specified sub array from the original array.
    """

    result = numpy.zeros(size, numpy.uint8)
    result_mask = numpy.ones(size, numpy.uint8)

    for i in range(size[0]):
        for j in range(size[1]):
            result[i][j] = array[(position[0] + i) % array.shape[0]][(position[1] + j) % array.shape[1]]

    return (result, result_mask)

   
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
def neighborhoodMatching(samples_value, neighborhood_value):

    samples = theano.tensor.constant(samples_value, 'samples')
    noise = theano.tensor.tensor3('noise', 'int32')

    noise_neighborhood = theano.tensor.tensor3('noise_neigh', 'int32')

    neighborhood = theano.tensor.constant(neighborhood_value, 'neighborhood')

    mismatch = (samples - noise) * noise_neighborhood * neighborhood

    output = theano.tensor.sum(mismatch ** 2, [2, 3, 4])
    best = theano.tensor.argmin(output)

    f = theano.function([noise, noise_neighborhood], best)
    return f
    
    
# Defines and returns a theano function for calculating the neighborhood match
# between a patch of noise and an array of sample patches
def neighborhoodMatching2(neighborhood_value):

    tensor5 = theano.tensor.TensorType('uint8', (False,)*5)

    samples = tensor5('samples')
    noise = theano.tensor.tensor3('noise', 'int16')

    noise_neighborhood = theano.tensor.tensor3('noise_neigh', 'int32')

    neighborhood = theano.tensor.constant(neighborhood_value, 'neighborhood')

    mismatch = (samples - noise) * noise_neighborhood * neighborhood

    output = theano.tensor.sum(mismatch ** 2, [2, 3, 4])
    best = theano.tensor.argmin(output)

    f = theano.function([samples, noise, noise_neighborhood], best)
    return f    


def synthesisTexture(sample, noise, neighborhood, sequential, wrap, patchSize,
                     mergeSize, mergeMode):

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

    noise_copy = numpy.copy(noise)

    # Presample every possible sub_array of the same size as neighborhood from the sample image.
    # It will save a LOT of time later              
    samples = numpy.asarray([[sample[i:i+neighborhood.shape[0],j:j+neighborhood.shape[1]]
                              for i in range(len(sample) - neighborhood.shape[0] + 1)]
                             for j in range(len(sample[0]) - neighborhood.shape[1] + 1)])
    
    # Compile the theano function for neighborhood matching
    #neighMatchFn = neighborhoodMatching(samples, neighborhood)
    neighMatchFn = neighborhoodMatching2(neighborhood)

    # Compute utility values for patch matching
    nb_row_noise = len(noise_copy)
    nb_col_noise = len(noise_copy[0])
    half_patch_size = (patchSize - 1) / 2

    # For every patch in the noise image
    row_noise = 0
    while row_noise < nb_row_noise:

        print row_noise * nb_row_noise, '/', nb_row_noise * nb_col_noise

        col_noise = 0
        while col_noise < nb_col_noise:

            # Obtain the neighborhood of the next patch, in the noise image,
            # for which to compute the new pixel values.
            noiseNeigh, noiseMask = arraySamplingOp(noise_copy,
                                                    (row_noise - neighborhood.shape[0] / 2 + half_patch_size,
                                                     col_noise - neighborhood.shape[1] / 2 + half_patch_size),
                                                    neighborhood.shape)

            # Find the nearest neighbor in the sample image
            from datetime import datetime
            start = datetime.now()
            
            #bestNeigh = neighMatchFn(noiseNeigh, noiseMask)
            bestNeigh = neighMatchFn(samples, noiseNeigh, noiseMask)
            
            end = datetime.now()
            print (end - start)

            bestSample = samples[bestNeigh / samples.shape[1],
                                 bestNeigh % samples.shape[1]]

            # Extract, from the best neighborhood, the best pixel values for the current patch.
            bestPatch = bestSample[neighborhood.shape[0]/2 - half_patch_size:
                                   neighborhood.shape[0]/2 + patchSize/2 + 1,
                                   neighborhood.shape[1]/2 - half_patch_size:
                                   neighborhood.shape[1]/2 + patchSize/2 + 1]

            bestPatch = bestPatch[:nb_row_noise - row_noise,
                                  :nb_col_noise - col_noise]

            # Merge the patch with the part of the texture that has already
            # been generated
            if mergeMode == "overwrite":
                # Copy the new pixel values in the current patch of the
                # texture being generated
                newPixelValues[row_noise:row_noise + bestPatch.shape[0],
                               col_noise:col_noise + bestPatch.shape[1]] = bestPatch

                # If sequential, also copy the pixel values in the noise array
                if sequential:

                    noise_copy[row_noise:row_noise + bestPatch.shape[0],
                               col_noise:col_noise + bestPatch.shape[1]] = bestPatch

            elif mergeMode == "graphcut":
                mergePatch(newPixelValues, seamMap, seamMapInfo, initMap,
                           sourceMap, bestPatch, row_noise, col_noise)

                if sequential:
                    noise_copy = numpy.copy(newPixelValues)

            # Increment col_noise
            col_noise += patchSize - mergeSize

        row_noise += patchSize - mergeSize

    # Display the pre-"improvement phase" quality of the generated texture
    quality = (seamMap.sum(), (seamMap ** 2).sum())
    print "Pre-improvement : " + str(quality)

    # Iteratively improve the generated texture
    if mergeMode == "graphcut":

        regionSize = patchSize
        improvementNeigh = numpy.ones((regionSize, regionSize, 1), 
                                      dtype='int32')

        for noImprovement in range(0):

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
                                          (worstRegionCoord[0] - neighborhood.shape[0] / 2 + half_patch_size,
                                           worstRegionCoord[1] - neighborhood.shape[1] / 2 + half_patch_size),
                                          neighborhood.shape)[0]
                                          
            # Find the best patch in the swatch
            bestNeigh = neighMatchFn(samples, worstRegion, improvementNeigh)

            bestSample = samples[bestNeigh / samples.shape[1],
                                 bestNeigh % samples.shape[1]]

            bestPatch = bestSample[neighborhood.shape[0] / 2 - half_patch_size:
                                   neighborhood.shape[0] / 2 + patchSize / 2 + 1,
                                   neighborhood.shape[1] / 2 - half_patch_size:
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
        g_aux = networkx.Graph()

        for edge in g.edges(data=True):                
            if (edge[2]['capacity'] - flow[edge[0]][edge[1]] >= 1e-10):
                g_aux.add_edge(edge[0], edge[1])

        # Partition the auxiliary graph according to which nodes are connected to
        # the source node and which are connected to the target node.
        if 'A' in g_aux:
            sourcePartition = set(networkx.single_source_shortest_path(g_aux, 'A'))
        else:
            sourcePartition = set()
            
        if 'B' in g_aux:
            targetPartition = set(networkx.single_source_shortest_path(g_aux, 'B'))
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
    else:
        #import pdb
        #pdb.set_trace()
        pass


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
    normalized_val = (val - minVal) / (maxVal - minVal)

    # Compute the corresponding color
    if normalized_val < 0.5:
        rgb[0] = int(255 * 2 * normalized_val)
        rgb[1] = 255
    else:
        rgb[0] = 255
        rgb[1] = 255 - int(255 * 2 * (normalized_val - 0.5))

    return rgb


def testTextureSynthesis(datasetName, textureName, textureSize, dictOrder,
                         dictWrap, dictNeighSize, dictNeighType, dictInput,
                         nbItt, dictPatchSize, dictNeighWeightInc,
                         dictMergeSize, dictMergeMode):

    # Generate a list of every possible combination of the parameters values
    # arrays
    param_combinations = list(itertools.product(dictOrder, dictWrap,
                                                dictNeighSize,
                                                dictNeighType, dictInput,
                                                dictPatchSize,
                                                dictNeighWeightInc,
                                                dictMergeSize,
                                                dictMergeMode))

    """
    # Remove from the list the combinations that are invalid, superfluous
    # or give very poor results.
    param_combinations = [p for p in param_combinations if not (p[0] == 'parallel')]

    param_combinations = [p for p in param_combinations if not (p[1] == 'wrap')]

    param_combinations = [p for p in param_combinations if not (p[3] == 'full' and
                                                                p[4] == 'fromNoise')]

    param_combinations = [p for p in param_combinations if not (p[3] == 'partial' and
                                                                p[4] == 'fromSample')]
    """

    # Run texture synthesis for every valid combination of the parameters
    # values
    for params in param_combinations:

        print "Starting test : " + "_".join(params)

        # Obtain the keys contained in params
        (orderKey, wrapKey, neighSizeKey, neighTypeKey,
         inputKey, patchSizeKey, neighIncKey, mergeSizeKey,
         mergeModeKey) = params

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

        # Load the texture sample from which to generate a new texture
        # and cast it as a numpy array
        if datasetName == "brodatz":
            imageFilename = "/data/lisa/data/Brodatz/" + textureName + ".gif"
        elif datasetName == "ubi":
            imageFilename = "/data/lisa/data/ubi/textures/" + textureName + ".jpg"

        image = Image.open(imageFilename)
        imgArray = numpy.asarray(image)

        # If grayscale, add a third dimension of size 1 to ensure consistency
        # between grayscale and color textures
        if len(imgArray.shape) == 2:
            imgArray = numpy.reshape(imgArray, imgArray.shape + (1,))
        nbColorChannels = imgArray.shape[2]

        # Load/generate the starting 'noise' for texture synthesis and cast it
        # as a numpy array
        noiseArrSize = 400
        noise_array = None
                
        if inputValue:
            # Generate input noise
            noise_array = numpy.random.randint(0, 256, (noiseArrSize, noiseArrSize, nbColorChannels))
            noise_array = numpy.asarray(noise_array, dtype='uint8')
        else:
            # Load a sample generated by the model
            noise_filename = "./Samples_FullScale/" + datasetName + "/" + textureName + "/sample.png"
            noise_image = Image.open(noise_filename)
            noise_array = numpy.asarray(noise_image)

            # If grayscale, add a third dimension of size 1 to ensure
            # consistency between grayscale and color textures
            if len(noise_array.shape) == 2:
                noise_array = numpy.reshape(noise_array, noise_array.shape + (1,))
            nbColorChannelsNoise = noise_array.shape[2]

            if nbColorChannels != nbColorChannelsNoise:
                if nbColorChannels == 1 and nbColorChannelsNoise > 1:
                    noise_array = numpy.mean(noise_array, 2)
                elif nbColorChannels > 1 and nbColorChannelsNoise == 1:
                    noise_array = numpy.reshape(noise_array,noise_array.shape()+(nbColorChannels,))
                else:
                    noise_array = numpy.mean(noise_array, 2)
                    imgArray = numpy.mean(imgArray, 2)

            noise_array = noise_array[0:noiseArrSize, 0:noiseArrSize]

        train_img_array = imgArray[:imgArray.shape[0] / 2, :]
        train_img_array = train_img_array[:textureSize, :textureSize]
        test_img_array = imgArray[imgArray.shape[0] / 2:, :]

        # Generate the neighborhood       
        neighborhood = generateWeightedNeighborhoodArray(neighSizeValue +
                                                         noise_array.shape[2:],
                                                         neighTypeValue,
                                                         neighIncValue,
                                                         patchSizeValue,
                                                         mergeSizeValue)

        # Perform texture synthesis and save the result in a file whose name
        # contains the name of the parameter keys for easy identification
        newTexture = noise_array
        
        for i in range(1, nbItt + 1):
            newTexture, extraData = synthesisTexture(train_img_array,
                                                     newTexture,
                                                     neighborhood, orderValue,
                                                     wrapValue,
                                                     patchSizeValue,
                                                     mergeSizeValue,
                                                     mergeModeValue)

        sourceMap = extraData[0]
        seamMap = extraData[1]
        seamMapInfo = extraData[2]

        # If images are grayscale, remove the last dimensions to allow scipy
        # to save them properly
        if newTexture.shape[2] == 1:
            newTexture = newTexture.mean(2)

        if train_img_array.shape[2] == 1:
            train_img_array = train_img_array.mean(2)

        if noise_array.shape[2] == 1:
            noise_array = noise_array.mean(2)

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
        scipy.misc.imsave(textureFilename, train_img_array)

        originalFilename = folderName + "/noise_" + '_'.join(params) + ".png"
        scipy.misc.imsave(originalFilename, noise_array)

        sourceMapFilename = folderName + "/sourceMap_" + '_'.join(params) + ".png"
        scipy.misc.imsave(sourceMapFilename, sourceMap)

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

    dictNeighSize = {'31x31': (31, 31)}
    dictInput = {'fromNoise': True}
    dictPatchSize = {'31x31patch': 31}
    dictMergeSize = {"mergeSize8": 8}
    dictMergeMode = {"graphcut": "graphcut"}
    dictNeighWeightInc = {"inc0": 0}

    # Perform texture synthesis with the previously created parameter
    # dictionaries
    testTextureSynthesis(datasetName, textureName, textureSize, dictOrder,
                         dictWrap, dictNeighSize, dictNeighType, dictInput,
                         nbItt, dictPatchSize, dictNeighWeightInc,
                         dictMergeSize, dictMergeMode)


def testTextureSynthesisSingle(datasetName, textureName, textureSize,
                               sequentialOrder, wrap, neighSize, fullNeigh,
                               fromNoise, nbItt, patchSize, neighWeightInc,
                               mergeSize, mergeMode):

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

    # Perform texture synthesis with the previously created parameter
    # dictionaries
    testTextureSynthesis(datasetName, textureName, textureSize, dictOrder,
                         dictWrap, dictNeighSize, dictNeighType, dictInput,
                         nbItt, dictPatchSize, dictNeighWeightInc,
                         dictMergeSize, dictMergeMode)


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

        # Perform texture synthesis with the given parameters
        testTextureSynthesisSingle(datasetName, textureName, textureSize,
                                   sequentialOrder, wrap, neighborhoodSize,
                                   fullNeighborhood, fromNoise, nbItt,
                                   patchSize, neighborhoodWeightInc,
                                   mergeSize, mergeMode)
