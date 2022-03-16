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
from collections import Counter

# There are 9 domains in the British National Corpus:
# Imaginative, natural & pure science, applied science, social science, world affairs, commerce & finance, arts, belief & thought, leisure
# The goal is to create a classifier to classify any piece of text into those domain. We also want a UNKNOWN domain.
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

        return message_set
        
    # TODO: Do freq analysis for each domain
    def RetriveMessageAndWordFromFile(self, path):
        domain_match = self.domain_finder.search(
            self.bnc_corpus_reader.raw(path))
        title = self.bnc_corpus_reader.xml(path)[0][0][0][0].text
        sents_in_message = self.bnc_corpus_reader.tagged_sents(fileids=path)
        # In British National Corpus, there is only one domain in one file. We ignore the files that don't have domain for the training set
        return sents_in_message, domain_match.group() if domain_match else 'unknown', title

    @staticmethod
    def CleanWordsList(words):
        return [word.lower() for word in words if word.lower() not in BNCDatasetReader.__IGNORED_WORD and word.isalpha()]

    __ROOT = '/Users/shihyaol/Downloads/ota_20.500.12024_2554/download/Texts'
    __IGNORED_WORD = stopwords.words('english') + list(string.punctuation)

#TODO: Tf-idf or Log Tf-idf
class FeaturesetCreater:
    def BagOfWordVectorization(self, message_set):
        vectorized_list = []
        word_dict = FeaturesetCreater.CreateCompleteWordDict(message_set)
        sentenses_to_domain_list = [(message['message']['sentenses'], message['metadata']['domain']) for message in message_set]
        for sentenses, label in sentenses_to_domain_list:
            vector_dict = dict.fromkeys(word_dict.keys(), 0)
            for sentense in sentenses:
                for word in sentense:
                    word_pos_tuple = (word[0], word[1])
                    if word_pos_tuple in word_dict:
                        vector_dict[word_pos_tuple] = 1
            vectorized_list.append((vector_dict, label))
        return vectorized_list

    def CreateBowVectorizedFeatureset(self, message_set, test_size=0.2):
        vectorized_message_set = self.BagOfWordVectorization(
            message_set)
        random.shuffle(vectorized_message_set)
        return vectorized_message_set[(int)(test_size * len(vectorized_message_set)):], vectorized_message_set[:(int)(test_size * len(vectorized_message_set))]

    @staticmethod
    def BalanceDataByOverSampling(message_set):
        message_dict = {}
        for message, domain in message_set:
            if domain in message_dict:
                message_dict[domain].append(message)
            else:
                message_dict[domain] = [message]
        max_domain_size = max(len(value) for value in message_dict.values())
        return [(m, k) for (k,v) in message_dict.items() for m in FeaturesetCreater.ExtendListByDuplicateValue(v, max_domain_size)]
    
    @staticmethod
    def ExtendListByDuplicateValue(list, max_domain_size):
        while len(list)*2 < max_domain_size:
            list.extend(list)
        random.shuffle(list)
        list.extend(list[0: max_domain_size - len(list)])
        return list

    @staticmethod
    def CreateCompleteWordDict(message_set):
        return FreqDist((word[0], word[1]) for message in message_set for sentense in message['message']['sentenses'] for word in sentense)      

def main():
    setup_module()
    # dataset_reader = BNCDatasetReader(file_range_regex='[A-D]\/\w+\/\w+\.xml')
    # message_set = dataset_reader.ReadFilesInputAndOutput()
    # with open('A_D.json', 'w') as output:
    #     json.dump(message_set, output)
    with open('A.json', 'r') as input:
        data = input.read()
    message_set = json.loads(data)
    featureset_creater = FeaturesetCreater()
    
    train, test = featureset_creater.CreateBowVectorizedFeatureset(
        message_set, test_size=0.15)
    print("Number of data points: {}".format(len(message_set)))
    classifier = NaiveBayesClassifier.train(train)

    print(classifier.show_most_informative_features())
    print("Domains: {}".format(classifier.labels()))
    print("Accuracy on testset: {}".format(
        classify.accuracy(classifier, test)))

if __name__ == '__main__':
    main()
