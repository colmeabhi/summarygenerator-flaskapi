import collections
import math
import re
import sys

import nltk

from collections import Counter

from .resources import (
    stop,
    porter,
    WORD,
    caps,
    prefixes,
    suffixes,
    starters,
    acronyms,
    websites,
)

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    #if "," in text: text = text.replace(",\"","\",")

    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    #text = text.replace(",","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def remove_stop_words(sentences) :
    tokenized_sentences = []
    for sentence in sentences :
        tokens = []
        split = sentence.lower().split()
        for word in split :
            if word not in stop :
                try :

                    tokens.append(porter.stem(word))
                except :
                    tokens.append(word)

        tokenized_sentences.append(tokens)
    return tokenized_sentences


def posTagger(tokenized_sentences) :
    tagged = []
    for sentence in tokenized_sentences :
        tag = nltk.pos_tag(sentence)
        tagged.append(tag)
    return tagged


def tfIsf(tokenized_sentences):
    scores = []
    COUNTS = []
    for sentence in tokenized_sentences :
        counts = collections.Counter(sentence)
        isf = []
        score = 0
        for word in counts.keys() :
            count_word = 1
            for sen in tokenized_sentences :
                for w in sen :
                    if word == w :
                        count_word += 1
            score = score + counts[word]*math.log(count_word-1)
        scores.append(score/len(sentence))
    return scores


def similar(tokens_a, tokens_b) :
    #Using Jaccard similarity to calculate if two sentences are similar
    ratio = len(set(tokens_a).intersection(tokens_b))/ float(len(set(tokens_a).union(tokens_b)))
    return ratio


def similarityScores(tokenized_sentences) :
    scores = []
    for sentence in tokenized_sentences :
        score = 0;
        for sen in tokenized_sentences :
            if sen != sentence :
                score += similar(sentence,sen)
        scores.append(score)
    return scores


def properNounScores(tagged) :
    scores = []
    for i in range(len(tagged)) :
        score = 0
        for j in range(len(tagged[i])) :
            if(tagged[i][j][1]== 'NNP' or tagged[i][j][1]=='NNPS') :
                score += 1
        scores.append(score/float(len(tagged[i])))
    return scores


def text_to_vector(text):
    words = WORD.findall(text)
    return collections.Counter(words)


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def centroidSimilarity(sentences,tfIsfScore) :
    centroidIndex = tfIsfScore.index(max(tfIsfScore))
    scores = []
    for sentence in sentences :
        vec1 = text_to_vector(sentences[centroidIndex])
        vec2 = text_to_vector(sentence)

        score = get_cosine(vec1,vec2)
        scores.append(score)
    return scores


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def numericToken(tokenized_sentences):
    scores = []
    for sentence in tokenized_sentences :
        score = 0
        for word in sentence :
            if is_number(word) :
                score +=1
        scores.append(score/float(len(sentence)))
    return scores




def sentencePos(sentences) :
    th = 0.2
    minv = th*len(sentences)
    maxv = th*2*len(sentences)
    pos = []
    for i in range(len(sentences)):
        if i==0 or i==len((sentences)):
            pos.append(0)
        else:
            t = math.cos((i-minv)*((1/maxv)-minv))
            pos.append(t)

    return pos


def sentenceLength(tokenized_sentences) :
    count = []
    maxLength = sys.maxsize
    for sentence in tokenized_sentences:
        num_words = 0
        for word in sentence :
                num_words +=1
        if num_words < 3 :
            count.append(0)
        else :
            count.append(num_words)

    count = [1.0*x/(maxLength) for x in count]
    return count


def thematicFeature(tokenized_sentences) :
    word_list = []
    for sentence in tokenized_sentences :
        for word in sentence :
            try:
                word = ''.join(e for e in word if e.isalnum())
                #print(word)
                word_list.append(word)
            except Exception as e:
                print("ERR")
    counts = Counter(word_list)
    number_of_words = len(counts)
    most_common = counts.most_common(10)
    thematic_words = []
    for data in most_common :
        thematic_words.append(data[0])
    print(thematic_words)
    scores = []
    for sentence in tokenized_sentences :
        score = 0
        for word in sentence :
            try:
                word = ''.join(e for e in word if e.isalnum())
                if word in thematic_words :
                    score = score + 1
                #print(word)
            except Exception as e:
                print("ERR")
        score = 1.0*score/(number_of_words)
        scores.append(score)
