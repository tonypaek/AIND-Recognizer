import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []


    # TODO implement the recognizer
    for i in range(test_set.num_items):
        word_log={}
        sequences, length = test_set.get_item_Xlengths(i)

        for w, m in models.items():
            try:
              score = m.score(sequences,length)
              word_log[w]=score
            except:
                word_log[w]=float("-inf")
        probabilities.append(word_log)
        best_word=max(word_log,key=word_log.get)
        guesses.append(best_word)
    return probabilities,guesses
    