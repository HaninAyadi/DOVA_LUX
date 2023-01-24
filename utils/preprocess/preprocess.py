"""

Preprocessing functions

Author: Adrian Ahne
Creation date: 24/04/2018

Editor: Hanin Ayadi
Last editing date: 21/07/2022

"""
import string
import unicodedata

import sys
from abc import abstractmethod

import inflect  # natural language related tasks of generating plurals, singular nouns, etc.
from nltk.tokenize import TweetTokenizer
import numpy as np

from defines import *
from contractions_def import *


# assume input matrix contains term frequencies
def tfidf_transform(mat):
    # convert matrix of counts to matrix of normalized frequencies
    normalized_mat = mat / np.transpose(mat.sum(axis=1)[np.newaxis])

    # compute IDF scores for each word given the corpus
    docs_using_terms = np.count_nonzero(mat, axis=0)
    idf_scores = np.log(mat.shape[1] / docs_using_terms)

    # compute tfidf scores
    tfidf_mat = normalized_mat * idf_scores
    return tfidf_mat


class Preprocess:

    def __init__(self, lang="english"):
        self.TweetTokenizer = TweetTokenizer()
        # Constant words like URL, USER, EMOT_SMILE, etc. that we want to keep in uppercase
        self.Constant_words = [value for attr, value in Constants.__dict__.items()
                               if not callable(getattr(Constants, attr)) and
                               not attr.startswith("__")] + Emotions.EMOTION_CATEGORIES

        self.WN_Lemmatizer_EN = WordNetLemmatizer()
        self.lang = lang

    @abstractmethod
    def get_text(self, raw_input):
        pass

    def replace_contractions(self, text):
        """ Replace contractions in string of text
            Examples:
              "aren't": "are not",
              "can't": "cannot",
              "'cause": "because",
              "hasn't": "has not",
              "he'll": "he will",

              FIXME: it occurs that
              - "people were in a hurry" is transformed to "we are in a hurry"  !!!
              - "are the main cause of obeisty" -> "are the main because of obesity"
              - "in the U.S are" -> "in the you.S. are"
        """

        if self.lang == "english":
            return contractions_fix(text)
        else:
            return text

    def replace_hashtags_URL_USER(self, text, mode_URL="keep",
                                  mode_Mentions="keep", mode_Hashtag="keep"):
        """
            Function handling the preprocessing of the hashtags, Mentions
            and URL patterns

            Parameters
            -------------------------------------------------------

            mode_URL : ("replace", "delete")
                       if "replace" : all url's in the text are replaced with the value of Constants.URL
                       if "delete" : all url's are deleted
                       if "keep" : keep url

            mode_Mentions : ("replace", "delete", "screen_name")
                       if "replace" : all user mentions in the text are replaced
                                      with the value of Constants.USER
                       if "delete" : all user mentions are deleted
                       if "screen_name" : delete '@' of all user mentions
                       if "keep" : keep user mention

            mode_Hashtag : ("replace", "delete")
                       if "replace" : all '#' from the hashtags are deleted
                       if "delete" : all hashtags are deleted
                       if 'keep' : keep hashtag

            Return
            -------------------------------------------------------------
            List of preprocessed text tokens

            https://github.com/yogeshg/Twitter-Sentiment

            Ex.:
            s = "@Obama loves #stackoverflow because #people are very #helpful!, \
                 check https://t.co/z2zdz1uYsd"
            print(replace_hashtags_URL_USER(s))
            >> "USER loves stackoverflow because people are very helpful!, check URL"


        """
        if mode_URL == "replace":
            text = Patterns.URL_PATTERN.sub(Constants.URL, text)
        elif mode_URL == "delete":
            text = Patterns.URL_PATTERN.sub("", text)
        elif mode_URL == "keep":
            text = text
        else:
            print("ERROR: mode_URL {} not defined!".format(mode_URL))
            exit()

        if mode_Mentions == "replace":
            text = Patterns.MENTION_PATTERN.sub(Constants.USER, text)
        elif mode_Mentions == "delete":
            text = Patterns.MENTION_PATTERN.sub("", text)
        elif mode_Mentions == "screen_name":
            mentions = Patterns.MENTION_PATTERN.findall(text)
            for mention in mentions:
                text = text.replace("@" + mention, mention)
        elif mode_Mentions == "keep":
            text = text
        else:
            print("ERROR: mode_Mentions {} not defined!".format(mode_Mentions))
            exit()

        if mode_Hashtag == "replace":
            hashtags = Patterns.HASHTAG_PATTERN.findall(text)
            for hashtag in hashtags:
                text = text.replace("#" + hashtag, hashtag)
        elif mode_Hashtag == "delete":
            hashtags = Patterns.HASHTAG_PATTERN.findall(text)
            for hashtag in hashtags:
                text = text.replace("#" + hashtag, "")
        elif mode_Hashtag == "keep":
            text = text
        else:
            print("ERROR: mode_Hashtag {} not defined!".format(mode_Hashtag))
            exit()

        return text

    def replace_special_words(self, text):
        """
            Replace special words

            For ex.: all the type 1 related words like "#type1", "Type 1", "t1d", etc.
                     are transformed to "type1"

        """

        # replace long covid words
        text = WordLists.LONGCOVID_WORDS.sub(Constants.LONGCOVID, text)

        # replace covid words
        text = WordLists.COVID_WORDS.sub(Constants.COVID, text)

        # replace long term words
        text = WordLists.LONGTERM_WORDS.sub(Constants.LONGTERM, text)

        return text

    def remove_repeating_characters(self, text):
        """
            If a word contains repeating characters, reduce it to only two repeating characters
            Ex. "coooooool" => "cool"
        """
        return re.sub(r'(.)\1+', r'\1\1', text)

    def remove_repeating_words(self, text):
        """
            Remove repeating words and only keep one
            Ex.: "I so need need need to sing" => "I so need to sing"
        """
        return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    def tokenize(self, text):
        """
            Tokenizes text in its single components (words, emojis, emoticons)

            Ex.:
            s = "I love:D python 😄 :-)"
            print(tokenize(s))
            >> ['I', 'love', ':D', 'python', '😄', ':-)']
        """
        return list(self.TweetTokenizer.tokenize(text))

    def remove_punctuation(self, text):
        """
            Remove punctuations from list of tokenized words

            Punctuations: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~...…

            TODO: check if !,? may contain useful information
        """

        def check_punctuation(word):
            return word not in string.punctuation and word not in ['...', '…', '..', "\n", "\t", " ", ""]

        return [word for word in text if check_punctuation(word)]

    def preprocess_emojis(self, text, limit_nEmojis=False):
        '''
            Replace emojis with their emotion category

            Parameters:
            ------------------------------------------------------------
            text:          tokenized text
            limit_nEmojis:  give maximum number of emojis of the same emotion category
                            that should occur in a text. Delete the other ones
                            Default: False, all emojis are considered

            Return
            ---------------------------------------------------------------
            tokenized text with replaced emojis by their emotion category
        '''

        if self.lang == "english":

            # counts occurrences of emojis in their emotion category
            emot_counter = {}
            for emotion in Emotions.EMOTION_CATEGORIES:
                emot_counter[emotion] = 0

            clean_text = []
            for ind, char in enumerate(text):
                if char in UNICODE_EMOJI:

                    emot_cat = EMOJI_TO_CATEGORY[UNICODE_EMOJI[char]]
                    if emot_cat != "":

                        if limit_nEmojis is not False:

                            # it is possible that one emoji is categorized into two
                            # different categories, for instance: 'EMOT_SURPRISE EMOT_FEAR'
                            emot_cat = emot_cat.split(" ")
                            for emo in emot_cat:
                                emot_counter[emo] += 1  # counts for the emotion in this text
                                if emot_counter[emo] <= limit_nEmojis:
                                    clean_text.append(emo)
                        else:
                            clean_text.append(emot_cat)

                    else:
                        print("INFO: No category set for emoji {} -> delete emoji {}".format(char, UNICODE_EMOJI[char]))
                else:
                    clean_text.append(char)

            return clean_text

        # other language
        else:
            return (text)

    def preprocess_emoticons(self, text):
        '''
            Replace emoticons in text with their emotion category by searching for
            emoticons with the pattern key word
        '''

        if self.lang == "english":
            clean_text = []
            for word in text:
                match_emoticon = Patterns.EMOTICONS_PATTERN.findall(word)
                if not match_emoticon:  # if no emoticon found
                    clean_text.append(word)
                else:
                    if match_emoticon[0] is not ':':
                        if match_emoticon[0] is not word:
                            clean_text.append(word)
                        else:
                            try:
                                clean_text.append(EMOTICONS[word])
                            except:
                                print("INFO: Could not replace emoticon: {} of the word: {}".format(match_emoticon[0],
                                                                                                    word),
                                      sys.exc_info())
            return clean_text

        # other languages
        else:
            return text

    def to_lowercase(self, text):
        """
            Convert all characters to lowercase from list of tokenized words
            Remark: Do it after emotion treatment, otherwise smiley :D -> :d
        """

        return [word.lower() if word not in self.Constant_words else word for word in text]

    def remove_non_ascii(self, text):
        """Remove non-ASCII characters from list of tokenized words"""

        return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') \
                for word in text if
                unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') is not ""]

    def replace_numbers(self, text, mode="replace"):
        """
            Replace all interger occurrences in list of tokenized words with textual representation
        """

        if mode == "replace":
            p = inflect.engine()
            return [p.number_to_words(word) if word.isdigit() else word for word in text]

        elif mode == "delete":
            return [word for word in text if not word.isdigit()]

    def remove_stopwords(self, text, include_personal_words=False, include_negations=False,
                         list_stopwords_manual=[]):
        """
            Remove stop words from list of tokenized words

            Parameter:
                text : tokenized list of strings
                include_personal_words : [True, False]
                                        if True, personal stopwords like
                                        "I", "me", "my" are not considered as
                                        stopwords
                include_negations: [True, False]
                                    if True, negation words like "no", "not" ,"nothing"
                                    are included and not considered as stopwords
                ignore_whitelist : whitelist containing words

                list_stopwords_manual : list with stopwords that overwrites the default stop lists if given

        """

        new_text = []

        # manual list of stopwords provided
        if len(list_stopwords_manual) > 0:
            return [word for word in text if word not in list_stopwords_manual]

        else:
            # english language
            if self.lang == "english":

                for word in text:

                    if include_personal_words:
                        if include_negations:
                            if (
                                    word not in Grammar.STOPWORDS_NO_PERSONAL_EN and word not in Grammar.STOPWORDS_CUSTOM) or word in Grammar.WHITELIST_EN:
                                new_text.append(word)
                        else:
                            if word not in Grammar.STOPWORDS_NO_PERSONAL_EN and word not in Grammar.STOPWORDS_CUSTOM:
                                new_text.append(word)
                    else:
                        if include_negations:
                            if (
                                    word not in Grammar.STOPWORDS_EN and word not in Grammar.STOPWORDS_CUSTOM) or word in Grammar.WHITELIST_EN:
                                new_text.append(word)
                        else:
                            if word not in Grammar.STOPWORDS_EN and word not in Grammar.STOPWORDS_CUSTOM:
                                new_text.append(word)
                return new_text

            # french language
            elif self.lang == "french":
                for word in text:
                    if word not in Grammar.STOPWORDS_FR:
                        new_text.append(word)
                return new_text

            # other languages
            else:
                return text

    def lemmatize_verbs(self, text):

        """ Lemmatize verbs in list of tokenized words
        """

        # Lemmatization
        def lookup_pos(pos):
            pos_first_char = pos[0].lower()
            if pos_first_char in 'nv':
                return pos_first_char
            else:
                return 'n'

        if self.lang == "english":

            # Part-of-speech tagging
            pos_tags = nltk.pos_tag(text)

            return [self.WN_Lemmatizer_EN.lemmatize(word, lookup_pos(pos)) for (word, pos) in pos_tags]

        else:
            return text

    def stem_words(self, text, stemmer=False):
        """ Stem words in list of tokenized words

            Parameter:
                - text :   tokenized list of words of the text

                - stemmer : algorithm to use for stemming, options:
                            - Grammar.STEMMER_SNOWBALL_EN
                            - Grammar.PORTER
                            - Grammar.STEMMER_LANCASTER
                            - Grammar.STEMMER_SNOWBALL_FR

            Remark: Three major stemming algorithms
                - Porter: most commonly used, oldest, most computationally expensive
                - Snowball / Porter2: better than Porter, a bit faster than Porter
                - Lancaster: aggressive algorithm, sometimes to a fault; fastest algo
                            often not intuitiive words; reduces words space hugely

                - Snowball French
        """

        for ind, word in enumerate(text):
            if word not in self.Constant_words:  # do not change words like USER, URL, EMOT_SMILE,...
                if stemmer != False:
                    text[ind] = stemmer.stem(word)
                elif self.lang == "english":
                    text[ind] = Grammar.STEMMER_SNOWBALL_EN.stem(word)
                elif self.lang == "french":
                    text[ind] = Grammar.STEMMER_SNOWBALL_FR.stem(word)

        return text


class PreprocesTwitter(Preprocess):

    # Overriding abstract method
    def get_text(self, raw_input):
        """ get text of text object in json format """
        return raw_input["text"]


class PreprocesReddit(Preprocess):

    # Overriding abstract method
    def get_text(self, raw_input):
        """ get text of text object in json format """
        return raw_input["fulltext"]