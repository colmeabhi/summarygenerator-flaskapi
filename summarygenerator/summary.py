import os

import numpy

from .entity import namedEntityRecog
# Use simple RBM replacement to avoid Theano compatibility issues
from .rbm_simple import test_rbm_simple as test_rbm
from .resources import cwd, sentenceLengths
from .text_features import (
    centroidSimilarity,
    numericToken,
    posTagger,
    properNounScores,
    remove_stop_words,
    sentenceLength,
    sentencePos,
    similarityScores,
    split_into_sentences,
    tfIsf,
    thematicFeature,
)



def executeForAFile(text) :

    print(text)
    sentences = split_into_sentences(text)
    text_len = len(sentences)
    sentenceLengths.append(text_len)

    tokenized_sentences = remove_stop_words(sentences)
    tagged = posTagger(remove_stop_words(sentences))

    tfIsfScore = tfIsf(tokenized_sentences)
    similarityScore = similarityScores(tokenized_sentences)

    print("\n\nProper Noun Score : \n")
    properNounScore = properNounScores(tagged)
    print(properNounScore)
    centroidSimilarityScore = centroidSimilarity(sentences,tfIsfScore)
    numericTokenScore = numericToken(tokenized_sentences)
    namedEntityRecogScore = namedEntityRecog(sentences)
    sentencePosScore = sentencePos(sentences)
    sentenceLengthScore = sentenceLength(tokenized_sentences)
    thematicFeatureScore = thematicFeature(tokenized_sentences)

    featureMatrix = []
    featureMatrix.append(thematicFeatureScore)
    featureMatrix.append(sentencePosScore)
    featureMatrix.append(sentenceLengthScore)
    featureMatrix.append(properNounScore)
    featureMatrix.append(numericTokenScore)
    featureMatrix.append(namedEntityRecogScore)
    featureMatrix.append(tfIsfScore)
    featureMatrix.append(centroidSimilarityScore)


    featureMat = numpy.zeros((len(sentences),8))
    for i in range(8) :
        for j in range(len(sentences)):
            featureMat[j][i] = featureMatrix[i][j]

    print("\n\n\nPrinting Feature Matrix : ")
    print(featureMat)
    print("\n\n\nPrinting Feature Matrix Normed : ")

    feature_sum = []

    for i in range(len(numpy.sum(featureMat,axis=1))) :
        feature_sum.append(numpy.sum(featureMat,axis=1)[i])

    print(featureMat)
    for i in range(len(sentences)):
        print(featureMat[i])


    # Use modern RBM implementation (scikit-learn based)
    temp = test_rbm(dataset = featureMat,learning_rate=0.1, training_epochs=14, batch_size=5,n_chains=5,
             n_hidden=8)

    print("\n\n")
    print(numpy.sum(temp, axis=1))

    enhanced_feature_sum = []
    enhanced_feature_sum2 = []

    for i in range(len(numpy.sum(temp,axis=1))) :
        enhanced_feature_sum.append([numpy.sum(temp,axis=1)[i],i])
        enhanced_feature_sum2.append(numpy.sum(temp,axis=1)[i])

    print(enhanced_feature_sum)
    print("\n\n\n")

    enhanced_feature_sum.sort(key=lambda x: x[0])
    print(enhanced_feature_sum)

    length_to_be_extracted = len(enhanced_feature_sum)//2

    print("\n\nThe text is : \n\n")
    for x in range(len(sentences)):
        print(sentences[x])

    print("\n\n\nExtracted sentences : \n\n\n")
    extracted_sentences = []
    extracted_sentences.append([sentences[0], 0])

    indeces_extracted = []
    indeces_extracted.append(0)

    for x in range(length_to_be_extracted) :
        if(enhanced_feature_sum[x][1] != 0) :
            extracted_sentences.append([sentences[enhanced_feature_sum[x][1]], enhanced_feature_sum[x][1]])
            indeces_extracted.append(enhanced_feature_sum[x][1])

    extracted_sentences.sort(key=lambda x: x[1])

    finalText = ""
    print("\n\n\nExtracted Final Text : \n\n\n")
    for i in range(len(extracted_sentences)):
        print("\n"+extracted_sentences[i][0])
        finalText = finalText + extracted_sentences[i][0]

    return finalText
