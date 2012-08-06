import numpy
from PIL import Image
import scipy
import theano
import itertools
import sys


def getOutOfBoundSubArray(array, position, size):

    """
    array    : original array from which to sample the sub-array
    position : tuple containing the coordinates of the top-left corner of the
               desired sub-array.
    size     : tuple containing the size of the desired sub-array.


    Returns the specified sub array from the original array.
    """

    result = numpy.zeros(size, numpy.int32)
    result_mask = numpy.zeros(size, numpy.int32)

    for i in range(size[0]):
        for j in range(size[1]):

            row = position[0] + i
            col = position[1] + j

            if row >= 0 and col >= 0 and row < array.shape[0] and col < array.shape[1]:
                result[i][j] = array[row][col]
                result_mask[i][j] = 1

    return (result, result_mask)


def getWrappingSubArray(array, position, size):

    """
    array    : original array from which to sample the sub-array
    position : tuple containing the coordinates of the top-left corner of the
               desired sub-array.
    size     : tuple containing the size of the desired sub-array.


    Returns the specified sub array from the original array.
    """

    result = numpy.zeros(size, numpy.int32)
    result_mask = numpy.ones(size, numpy.int32)

    for i in range(size[0]):
        for j in range(size[1]):
            result[i][j] = array[(position[0] + i) % array.shape[0]][(position[1] + j) % array.shape[1]]

    return (result, result_mask)


def generateNeighborhoodArray(size, full, weightDist, inc):
    
    result = numpy.zeros(size, numpy.int32)
    
    if weightDist == "equal":

        if full:
            result =  numpy.ones(size, numpy.int32)
        else:
            result = numpy.zeros(size, numpy.int32)
            result[:size[0]/2] += 1
            result[size[0]/2, :size[1]/2] += 1
            
    elif weightDist == "center":
        
        if full:
            result =  numpy.ones(size, numpy.int32)
        else:
            result = numpy.zeros(size, numpy.int32)
            result[:size[0]/2] += 1
            result[size[0]/2, :size[1]/2] += 1
            
        for i in range(min(size[0] / 2, size[1] / 2)):
            center = result[size[0]/2 - i:size[0]/2 + i + 1, 
                            size[1]/2 - i:size[1]/2 + i + 1]
            
            result[size[0]/2 - i:size[0]/2 + i + 1, 
                   size[1]/2 - i:size[1]/2 + i + 1] += inc * (center != 0)
            
    elif weightDist == "border":
        
        if full:
            result =  numpy.ones(size, numpy.int32)
        else:
            result = numpy.zeros(size, numpy.int32)
            result[:size[0]/2] += 1
            result[size[0]/2, :size[1]/2] += 1
            
        for i in range(max(size[0] / 2, size[1] / 2)):
            
            center = result[size[0]/2 - i:size[0]/2 + i + 1, 
                            size[1]/2 - i:size[1]/2 + i + 1]
            
            result[size[0]/2 - i:size[0]/2 + i + 1, 
                   size[1]/2 - i:size[1]/2 + i + 1] -= inc * (center != 0)
                   
            result += inc
            
    return result
    

# Defines and returns a theano function for calculating the neighborhood match
# between a patch of noise and an array of sample patches
def neighborhoodMatching():
    samples = theano.tensor.tensor4('samples', dtype='int32')
    noise = theano.tensor.matrix('noise', 'int32')

    samples_neighborhood = theano.tensor.tensor4('samples_neigh', dtype='int32')
    noise_neighborhood = theano.tensor.matrix('noise_neigh', 'int32')
    neighborhood = theano.tensor.matrix('neighborhood', 'int32')

    mismatch = (samples - noise) * samples_neighborhood * noise_neighborhood * neighborhood
    
    output = theano.tensor.sum(mismatch ** 2, [2, 3])
    best = theano.tensor.argmin(output) 
    value = samples[best / samples.shape[1]][best % samples.shape[1]][samples.shape[2] / 2][samples.shape[3] / 2]
    
    f = theano.function([samples, noise, samples_neighborhood, noise_neighborhood, neighborhood], [output, best, value])
    return f
    
   
    
"""
def theanoSynthesisTexture(
    samples = theano.tensor.tensor4('samples', dtype='int32')
    noise = theano.tensor.matrix('noise', 'int32')
    neighborhood = theano.tensor.tensor4('samples_neigh', dtype='int32')
    sequential = theano.tensor.iscalar('sequential')
    wrap = theano.tensor.iscalar('wrap')
    
    newPixelValues = theano.tensor.matrix('noise', 'int32')
    noise_copy = theano.tensor.matrix('noise', 'int32')
    
    # TODO
    results, updates = theano.scan(fn=theanoSynthesisTextureRow,
                                   outputs_info=None,
                                   sequences=[noise],
                                   non_sequences=[samples, neighborhood,
                                                  sequential, wrap])
                                                  
                                                  
                

def theanoSynthesisTextureRow():
    return 1
    
    
def theanoSynthesisTexturePixel():
    return 1
"""
    


def synthesisTexture(sample, noise, neighborhood, sequential, wrap):

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
        noiseSamplingOp = getWrappingSubArray
        sampleSamplingOp = getOutOfBoundSubArray
    else:
        arraySamplingOp = getOutOfBoundSubArray
        noiseSamplingOp = getOutOfBoundSubArray
        sampleSamplingOp = getOutOfBoundSubArray
        
    newPixelValues = numpy.zeros(noise.shape, numpy.int32)
    noise_copy = numpy.copy(noise)
    
    neighMatchFn = neighborhoodMatching()

    # Presample every possible sub_array of the same size as neighborhood from the samble image.
    # It will save a LOT of time later   
    samples = numpy.asarray([[sampleSamplingOp(sample,(i,j),neighborhood.shape)[0]
                              for i in range(len(sample) - neighborhood.shape[0] + 1)] 
                             for j in range(len(sample[0]) - neighborhood.shape[1] + 1)])  
                                  
    samplesMask = numpy.asarray([[sampleSamplingOp(sample,(i,j),neighborhood.shape)[1]
                              for i in range(len(sample) - neighborhood.shape[0] + 1)] 
                             for j in range(len(sample[0]) - neighborhood.shape[1] + 1)]) 

    nb_row_noise = len(noise_copy)
    nb_col_noise = len(noise_copy[0])
    nb_row_sample = len(sample)
    nb_col_sample = len(sample[0]) 


    # For every pixel in the noise image
    for row_noise in range(nb_row_noise):
        
        print row_noise * nb_row_noise, '/', nb_row_noise * nb_col_noise
        
        for col_noise in range(nb_col_noise):

            # Obtain the neighborhood of the next pixel, in the noise image,
            # for which to compute its new value.
            noiseNeigh, noiseMask = arraySamplingOp(noise_copy,
                                                    (row_noise - neighborhood.shape[0]/2,
                                                     col_noise - neighborhood.shape[1]/2),
                                                    neighborhood.shape)

            # Find the nearest neighbor in the sample image
            theanoResult, bestNeigh, pixelValue = neighMatchFn(samples, noiseNeigh, samplesMask, noiseMask, neighborhood)
            
            # Store the value of the central pixel of the nearest neighbor
            newPixelValues[row_noise][col_noise] = pixelValue

            # If sequential, assign the value of the nearest neighbor to the pixel in noise
            if sequential:
                noise_copy[row_noise][col_noise] = pixelValue

    return newPixelValues;



def testTextureSynthesis(textureName, textureSize, dictOrder, dictWrap,
                         dictNeighSize, dictNeighType, dictInput, nbItt,
                         dictNeighWeightDist, dictNeighWeightInc):

    # Parameters values arrays
    """
    dictOrder = {'parallel': False, 'sequential': True}
    dictWrap  = {'wrap': True, 'nowrap': False}
    dictNeighSize = {'3x3': (3,3), '5x5': (5,5), '7x7': (7,7), '9x9': (9,9), '15x15': (15,15), '15x15': (19,19)}
    dictNeighType = {'full': True, 'partial': False}
    dictInput = {'fromNoise': True, 'fromSample' : False}
    nbItt = 1
    """
    
    # Generate a list of every possible combination of the parameters values
    param_combinations = list(itertools.product(dictOrder, dictWrap, dictNeighSize,
                                                dictNeighType, dictInput))
    
    # Remove from the list the combinations that are invalid, superfluous or give very poor results.    
    param_combinations = [p for p in param_combinations if not (p[1] == 'wrap')]
    param_combinations = [p for p in param_combinations if not (p[0] == 'parallel' and 
                                                                p[4] == 'fromNoise' )]
    param_combinations = [p for p in param_combinations if not (p[3] == 'full' and 
                                                                p[4] == 'fromNoise' )]
    param_combinations = [p for p in param_combinations if not (p[0] == 'sequential' and 
                                                                p[3] == 'partial' and
                                                                p[4] == 'fromSample' )]                                                         
                                                             
    
    
    # Run texture synthesis for every valid combination of the parameters values
    for params in param_combinations:
        
        print "Starting test : " + "_".join(params)

        # Obtain the keys contained in params
        (orderKey, wrapKey, neighSizeKey, neighTypeKey, inputKey) = params

        # Obtain the parameter values associated with the keys
        orderValue = dictOrder[orderKey]
        wrapValue = dictWrap[wrapKey]
        neighSizeValue = dictNeighSize[neighSizeKey]
        neighTypeValue = dictNeighType[neighTypeKey]
        inputValue = dictInput[inputKey]


        # Load the texture sample from which to generate a new texture
        # and cast it as a numpy array
        image_filename = "/data/lisa/data/Brodatz/" + str(textureName) + ".gif"
        image = Image.open(image_filename)
        image = image.resize((textureSize,textureSize), Image.BICUBIC)
        img_array = numpy.asarray(image)
               
        train_img_array = img_array[:textureSize/2,:]
        test_img_array = img_array[textureSize/2:,:]


        # Load/generate the starting 'noise' for texture synthesis and cast it as
        # a numpy array
        noise_array = None
        if inputValue:
            # Generate input noise
            noise_array = numpy.random.randint(0, 256, (200, 200))
        else:
            # Load a sample generated by the model
            noise_filename = "./Samples_FullScale/" + str(textureName) + "/sample.png"
            noise_image = Image.open(noise_filename)
            noise_array = numpy.asarray(noise_image)
            noise_array = noise_array[0:200, 0:200]
            
        # Generate the neighborhood
        neighborhood = generateNeighborhoodArray(neighSizeValue, 
                                                 neighTypeValue)


        # Perform texture synthesis and save the result in a file whose name contains
        # the name of the parameter keys for easy identification
        newTexture = noise_array
        for i in range(1,nbItt+1):
            newTexture = synthesisTexture(train_img_array, newTexture, neighborhood, orderValue, wrapValue)
        
        
        newTextureFilename = "./results/" + textureName + "/output_" + '_'.join(params) + ".jpg"
        scipy.misc.imsave(newTextureFilename, newTexture)
 
        # Save the sample used for generation
        originalFilename = "./results/" + textureName + "/texture_" + '_'.join(params) + ".jpg"
        scipy.misc.imsave(originalFilename, train_img_array)
        
        # Save the original texture sample in a file for comparison
        originalFilename = "./results/" + textureName + "/noise_" + '_'.join(params) + ".jpg"
        scipy.misc.imsave(originalFilename, noise_array)
        


def testTextureSynthesisMulti(textureName, textureSize):
    
    # Instantiate parameter dictionaries
    dictOrder = {'parallel': False, 'sequential': True}
    dictWrap  = {'wrap': True, 'nowrap': False}
    dictNeighSize = {'5x5': (5,5), '9x9': (9,9), '15x15': (15,15)}
    dictNeighType = {'full': True, 'partial': False}
    dictInput = {'fromNoise': True, 'fromSample' : False}
    nbItt = 1
    
    # Perform texture synthesis with the previously created parameter dictionaries
    testTextureSynthesis(textureName, textureSize, dictOrder, dictWrap,
                         dictNeighSize, dictNeighType, dictInput, nbItt)
 
 
def testTextureSynthesisSingle(textureName, textureSize, sequentialOrder, 
                               wrap, neighSize, fullNeigh, fromNoise, nbItt):
    
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
        
    
    # Perform texture synthesis with the previously created parameter dictionaries
    testTextureSynthesis(textureName, textureSize, dictOrder, dictWrap,
                         dictNeighSize, dictNeighType, dictInput, nbItt)
    
    
if __name__ == "__main__":
    
    if sys.argv[1] == 'multi':
        
        # Extract input parameters
        textureName = sys.argv[2]
        textureSize = int(sys.argv[3])
        
        # Perform texture synthesis with the given parameters
        testTextureSynthesisMulti(textureName, textureSize)
        
    if sys.argv[1] == 'single':
        
        # Extract input parameters
        textureName = sys.argv[2]
        textureSize = int(sys.argv[3])
        sequentialOrder = int(sys.argv[4]) != 0
        wrap = int(sys.argv[5]) != 0
        neighborhoodSize = int(sys.argv[6])
        fullNeighborhood = int(sys.argv[7]) != 0
        fromNoise = int(sys.argv[8]) != 0
        nbItt = int(sys.argv[9])
        
        # Perform texture synthesis with the given parameters
        testTextureSynthesisSingle(textureName, textureSize, sequentialOrder,
                                   wrap, neighborhoodSize, fullNeighborhood,
                                   fromNoise, nbItt, neighborhoodWeightDist, 
                                   neighborhoodWeightInc)
