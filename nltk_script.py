from importlib.metadata import metadata
import re
import os
import random
import string
import json
from nltk.corpus import stopwords
from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk.test.classify_fixt import setup_module
from nltk.probability import FreqDist
from nltk import NaiveBayesClassifier
from nltk import classify
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

# There are 9 domains in the British National Corpus:
# Imaginative, natural & pure science, applied science, social science, world affairs, commerce & finance, arts, belief & thought, leisure
# The goal is to create a classifier to classify any piece of text into those domain. We also want a UNKNOWN domain.
class Word:
    def __init__(self, str):
        str_split = str.split(' ')
        self.word_str = str_split[0]
        self.type = str_split[1]
    
    def isVerb(self):
        return self.type.startswith("V")
    
    def isNoun(self):
        return self.type.startswith("N")
    
    def isAdj(self):
        return self.type.startswith("AJ")
    
    def isArticle(self):
        return self.type == "AT0"
    
    def isAdverb(self):
        return self.type.startswith("AV")
    
    def isConjunction(self):
        return self.type.startswith("CJ")
    
    def isNumberOrNumeral(self):
        return self.type == "CRD" | self.type == "ORD"

    def isDeterminer(self):
        return self.type.startswith("DT")   
    
    def isPronoun(self):
        return self.type.startswith("PN") 
    
    def isOpenClass(self):
        return self.isVerb() or self.isAdj() or self.isNoun() or self.isAdverb()
    
    def isCloseClass(self):
        return not self.isOpenClass()
    
    def __str__(self):
        return self.word_str + " / " + self.type
    
    def __repr__(self):
        return self.word_str + " / " + self.type
    
class JsonDecoder:
    def DecodeSents(message_set):
        for i, message in enumerate(message_set):
            data = []
            for sent in message['message']['sentenses']:
                word = ""
                isTypeAdded = False
                for c in sent.split(' '):
                    word += c
                    if not isTypeAdded:
                        word += " "
                        isTypeAdded = True
                    else:
                        word_obj = Word(word)
                        data.append(word_obj)
                        isTypeAdded = False
                        word = ""
            message_set[i]['message']['sentenses'] = data
        return message_set
        
class BNCDatasetReader:
    def __init__(self, file_range_regex):
        self.domain_finder = re.compile("(?<=domain:\s)[a-zA-Z0-9_& ]+(?=\))")
        self.xml_finder = re.compile(file_range_regex)
        self.bnc_corpus_reader = BNCCorpusReader(
            root=BNCDatasetReader.__ROOT, fileids=file_range_regex, lazy=False)

    def ReadFilesInputAndOutput(self, dir_path=""):
        message_set = []
        dir_path_absolute = os.path.join(BNCDatasetReader.__ROOT, dir_path)

        for filename in os.listdir(dir_path_absolute):
            file_path = os.path.join(dir_path, filename)
            file_path_absolute = os.path.join(dir_path_absolute, filename)

            if os.path.isdir(file_path_absolute):
                messages_from_sub_folder = self.ReadFilesInputAndOutput(
                    file_path)
                if (messages_from_sub_folder):
                    message_set.extend(messages_from_sub_folder)
            elif self.xml_finder.match(file_path) and os.path.isfile(file_path_absolute):
                sents_in_message, domain, title = self.RetriveMessageAndWordFromFile(
                    file_path)
                metadata = {'domain': domain,'title': title, 'fileName': file_path}
                message = {'sentenses': sents_in_message}
                message_set.append({'metadata': metadata, 'message': message}) 
            
            if os.path.isdir(file_path_absolute) and len(filename) == 2:
                new_file = filename[0] + "/" + filename +'.json'
                os.makedirs(os.path.dirname(new_file), exist_ok=True)
                with open(new_file, 'w') as output:
                        json.dump(message_set, output, indent=4)
                message_set = []
        return message_set
        
    # TODO: Do freq analysis for each domain
    def RetriveMessageAndWordFromFile(self, path):
        domain_match = self.domain_finder.search(
            self.bnc_corpus_reader.raw(path))
        title = self.bnc_corpus_reader.xml(path)[0][0][0][0].text.strip().split('.')[0]
        sents_in_message = self.bnc_corpus_reader.tagged_sents(fileids=path, c5=True)
        merged_sents = []
        for sent in sents_in_message:
            merged_sent = ""
            trailing_space = ""
            for word in sent:
                merged_sent += trailing_space
                merged_sent += word[0] + " " + word[1]
                trailing_space = " "
            merged_sents.append(merged_sent)
                
        # In British National Corpus, there is only one domain in one file. We ignore the files that don't have domain for the training set
        return merged_sents, domain_match.group() if (domain_match and domain_match.group() == 'social science') else 'unknown', title

    @staticmethod
    def CleanWordsList(words):
        return [word.lower() for word in words if word.lower() not in BNCDatasetReader.__IGNORED_WORD and word.isalpha()]

    __ROOT = '/Users/shihyaol/Downloads/ota_20.500.12024_2554/download/Texts'
    __IGNORED_WORD = stopwords.words('english') + list(string.punctuation)

#TODO: Tf-idf or Log Tf-idf
class FeaturesetCreater:
    def BagOfWordVectorization(self, message_sets, filter_close_class):
        vectorized_list = []
        word_dict = dict(Counter([word.word_str for message_set in message_sets for m in message_set for word in m['message']['sentenses']]))
        sentense_to_domain_list = [(message['message']['sentenses'], message['metadata']['domain']) for message_set in message_sets for message in message_set]
        for sentense, label in sentense_to_domain_list:
            vector_dict = dict.fromkeys(word_dict.keys(), 0)
            for word in sentense:
                if filter_close_class and word.isCloseClass(): 
                    continue
                if word.word_str in word_dict:
                    vector_dict[word.word_str] = 1
            vectorized_list.append((vector_dict, label))
        return vectorized_list
    
    def BagOfTypeVectorization(self, message_sets, filter_close_class):
        vectorized_list = []
        word_type_dict = dict(Counter([word.type for message_set in message_sets for m in message_set for word in m['message']['sentenses']]))
        sentenses_to_domain_list = [(message['message']['sentenses'], message['metadata']['domain']) for message_set in message_sets for message in message_set]
        for sentense, label in sentenses_to_domain_list:
            vector_dict = dict.fromkeys(word_type_dict.keys(), 0)
            for word in sentense:
                if filter_close_class and word.isCloseClass(): 
                        continue
                word_type = word.type 
                if word_type in word_type_dict:
                    vector_dict[word_type] = 1
            vectorized_list.append((vector_dict, label))
        return vectorized_list

    def CreateBowVectorizedFeatureset(self, message_set, test_size=0.2, filter_close_class=False):
        vectorized_message_set = self.BagOfWordVectorization(
            message_set, filter_close_class)
        random.shuffle(vectorized_message_set)
        return vectorized_message_set[(int)(test_size * len(vectorized_message_set)):], vectorized_message_set[:(int)(test_size * len(vectorized_message_set))]

    def CreateBagOfTypeVectorizedFeatureset(self, message_sets, test_size=0.2, filter_close_class=False):
        vectorized_message_set = self.BagOfTypeVectorization(
            message_sets, filter_close_class)
        print(vectorized_message_set)
        random.shuffle(vectorized_message_set)
        return vectorized_message_set[(int)(test_size * len(vectorized_message_set)):], vectorized_message_set[:(int)(test_size * len(vectorized_message_set))]

    @staticmethod
    def ExtendListByDuplicateValue(list, max_domain_size):
        while len(list)*2 < max_domain_size:
            list.extend(list)
        random.shuffle(list)
        list.extend(list[0: max_domain_size - len(list)])
        return list
    
    @staticmethod
    def ExtractFeature(sentense, posPattern):
        found_pat_list = []
        for i in range(len(sentense)):
            if (sentense[i].type.startswith(posPattern[0])):
                found_pat = [sentense[i]]
                j = i + 1
                while j < len(sentense) and j - i < len(posPattern):
                    if not sentense[j].type.startswith(posPattern[j - i]):
                        break
                    else:
                        found_pat.append(sentense[j])
                    j += 1
                if j - i == len(posPattern):
                    found_pat_list.append(found_pat)
        return found_pat_list
    
    @staticmethod
    def PlotDist(data):
        count = dict(Counter(data).most_common(100))
        df = pd.DataFrame.from_dict(count, orient='index', columns=['Count']).sort_values(by='Count', ascending=False)
        df.plot(kind='bar')
        plt.show()
    
def main():
    setup_module()
    # dataset_reader = BNCDatasetReader(file_range_regex='[A-D]/\w+/\w+\.xml')
    # dataset_reader.ReadFilesInputAndOutput()
    dir_name = ['A','B','C','D','E','F','G','H','J','K']
    message_sets = []
    for dir in dir_name:
        for filename in os.listdir(dir):
            with open(dir + "/" + filename, 'r') as input:
                data = input.read()
                message_sets.append(JsonDecoder.DecodeSents(json.loads(data)))
    # bigram_type_list = [(m['message']['sentenses'][index].type + "/" + m['message']['sentenses'][index+1].type) for m in message_set for index, word in enumerate(m['message']['sentenses']) if index < (len(m['message']['sentenses']) - 1)]
    # type_list = [word.type for m in message_set for word in m['message']['sentenses']]
    # FeaturesetCreater.PlotDist(type_list)
    
    featureset_creater = FeaturesetCreater()
    
    train, test = featureset_creater.CreateBowVectorizedFeatureset(
        message_sets, test_size=0.15, filter_close_class=True)
    print("Number of data points: {}".format(len(message_sets)))
    classifier = NaiveBayesClassifier.train(train)

    print(classifier.show_most_informative_features())
    print("Domains: {}".format(classifier.labels()))
    print("Accuracy on testset: {}".format(
        classify.accuracy(classifier, test)))

if __name__ == '__main__':
    main()
