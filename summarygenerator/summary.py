import os

import numpy

from .entity import namedEntityRecog
from .rbm import test_rbm
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
