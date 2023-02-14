"""
Functions to preprocess text data.
Note:
Many other operation can be
performed on Gensim Word2vec (word
embedding) models. Refer to the
Gensim documentation for more information.
ref for gensim tokenize:
https://radimrehurek.com/gensim/utils.html#gensim.utils.tokenize
"""

from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import strip_punctuation
from gensim.utils import tokenize
from fuzzywuzzy import fuzz
import spacy
import pandas as pd
import numpy as np
import io


def load_stopwords(filename):
    """Returns stopwords in form of a list of string.

    :param filename: name of or path to the file containing the stopwords
    :return: list of stopwords
    """
    fid = io.open(filename, mode="r", encoding="utf-8")
    words = []
    for line in fid.readlines():
        words.append(line.replace("\n", ""))
    return words


def safe_tokenize(text, **kwargs):
    """Tokenize the input text using the tokenize() function, None inputs are allowed and returned as an empty string.

    :param text: string or None input
    :param \**kwargs: additional parameters passed to the Gensim tokenize function
    :return: tokenized string
    """
    if text is None:
        text = ""
    # remove punctuation (in particular underscores)
    text = strip_punctuation(text)
    return tokenize(text, **kwargs)


def tokenize_text(text, stopwords=None):
    """Tokenize the input text using the tokenize() function.

    :param text: input text string
    :param stopwords: list of stopwords whose corresponding tokens will be discarded
    :return: list of tokens where non-valid (untokenizable) data has been replaced by empty strings
    """
    tokens = safe_tokenize(
        text, lowercase=True, deacc=True, encoding="utf8", errors="strict"
    )
    if stopwords is not None:
        return ["" if tok in stopwords else tok for tok in tokens]
    return [tok for tok in tokens]


def unique_tokens(text, stopwords=None, return_counts=False):
    """Returns an array of unique tokens from an input text string.

    :param text: input text string
    :param stopwords: list of stopwords to exclude from the output
    :param return_counts: boolean flag (default False) to return the count of each unique token occurrence
    :return: array of unique tokens
    """
    return np.unique(tokenize_text(text, stopwords), return_counts=return_counts)


def normalize_text(text, unique=False, stopwords=None):
    """Returns a normalized text string, i.e. where all text has been standardized (lowercased, deaccentuated, and
    tokenized).

    :param text: input text string
    :param unique: boolean flag (default False), if set to True constructs the output string with unique tokens only
    :param stopwords: list of stopwords to exclude from the output
    :return: normalized text string
    """
    if unique:
        return " ".join(unique_tokens(text, stopwords))
    return " ".join(tokenize_text(text, stopwords))


def normalize_df_column(data, column, unique=False, stopwords=None):
    """Apply the normalize_text() function to an entire column of a dataframe.

    :param data: input dataframe
    :param column: column name of the dataframe containing the text to be normalized
    :param unique: boolean flag (default False), if set to True constructs new strings with unique tokens only
    :param stopwords: list of stopwords to exclude from the output
    :return: the transformed dataframe
    """
    if isinstance(data, pd.core.series.Series):
        return normalize_text(data[column], unique=unique, stopwords=stopwords)
    # data is of type pd.core.frame.DataFrame
    return data.apply(
        lambda x: normalize_text(x[column], unique=unique, stopwords=stopwords), axis=1
    )


def fuzzysearch_indices(data, column, string, level):
    """Returns boolean indicators where a string was matched in a given dataframe column. The level of fuzziness can
    be set using the 'level' variable. Fuzzysearch is based on the fuzzywuzzy package that computes string similarity
    using the Levenshtein distance.

    :param data: input dataframe
    :param column: column name of the dataframe containing the text data
    :param string: target string to match
    :param level: minimum level of fuzzyness accepted for a positive match (100 = very strict, 0 = very fuzzy matching)
    :return: boolean indicators where the string was matched
    """
    assert 0 <= level <= 100
    ratio = data[column].map(lambda s: fuzz.token_set_ratio(string, s))
    return ratio >= level


def fuzzysearch(data, column, string, level):
    """Returns rows of the input dataframe where a string was matched in a given dataframe column. The level of
    fuzziness can be set using the 'level' variable. Fuzzysearch is based on the fuzzywuzzy package that computes
    string similarity using the Levenshtein distance.

    :param data: input dataframe
    :param column: column name of the dataframe containing the text data
    :param string: target string to match
    :param level: minimum level of fuzzyness accepted for a positive match (100 = very strict, 0 = very fuzzy matching)
    :return: rows of the input dataframe where the string was matched
    """
    ind = fuzzysearch_indices(data, column, string, level)
    return data.loc[ind]


def replace_with_stem(
    data, column, stem, words_to_replace, strict=False, fuzzylevel=None
):
    """**Inplace** replacement of the 'words_to_replace' with the stem word in the column of interest
    if strict=True, the replacement will occur only if the text matches exactly one of the words

    :param data: input dataframe
    :param column: column name of the dataframe containing the text data that wil be replaced
    :param stem: stem (string) that will be used as a replacement
    :param words_to_replace: words that, when matched, will be replaced by the stem
    :param strict: boolean flag (default to False) if True, the replacement will occur only if the text matches exactly one of the words in 'words_to_replace'
    :param fuzzylevel: minimum level of fuzzyness accepted for a positive match (100 = very strict, 0 = very fuzzy matching)
    :return: None
    """
    # avoid None values
    data[column].fillna("", inplace=True)
    if strict:
        ind = data[column].apply(
            lambda x: np.sum([x == w for w in words_to_replace]) > 0
        )
    elif fuzzylevel is not None:
        ind = np.zeros(data.shape[0], dtype=np.bool)
        for word in words_to_replace:
            idx = fuzzysearch_indices(data, column, word, fuzzylevel)
            ind = np.logical_or(ind, idx)
    else:
        ind = data[column].apply(
            lambda x: np.sum([x.find(w) >= 0 for w in words_to_replace]) > 0
        )
    print(f'{np.sum(ind)} occurences replaced by the stem word "{stem}"')
    data.loc[ind, column] = stem
    return None


def replace_with_lemma(text, spacy_model=None, as_list=False):
    """Returns the input text where all of the tokens have been replaced by their lemma (i.e. word root). By default
    this function will use the spacy french 'fr_core_news_md' model.

    :param text: input raw text (should not be tokenized)
    :param spacy_model: spacy model used to compute the lemmas
    :param as_list: boolean flag (defaults to False), if True, returns the lemmas as a list instead of a string
    :return: lemmatized text
    """
    if spacy_model is None:
        print(
            "WARNING: no spacy model was provided, loading default model: 'fr_core_news_md'"
        )
        spacy_model = spacy.load("fr_core_news_md", disable=["parser", "ner"])
    nlp = spacy_model(text, disable=["parser", "ner"])
    tokens = [token.lemma_ for token in nlp]
    if as_list:
        return tokens
    return " ".join(tokens)


def create_bow_from_text(text, min_freq=0, max_freq=1, stopwords=None):
    """Returns a bag-of-word (BOW) object from a string of text using the gensim library.

    :param text: input text string
    :param min_freq: minimum word frequency below which tokens will be discarded from the BOW
    :param max_freq: maximum word frequency above which tokens will be discarded from the BOW
    :param stopwords: list of stopwords to exclude from the output
    :return: BOW (Dictionary gensim object)
    """
    assert 0 <= min_freq <= 1
    assert 0 <= max_freq <= 1
    text = normalize_text(text, stopwords=stopwords)
    tokens, counts = unique_tokens(text, return_counts=True)
    order = np.argsort(counts)[::-1]
    tokens = tokens[order]
    counts = counts[order] / np.max(counts)
    # remove most frequent and infrequent words
    ind = np.logical_and(counts >= min_freq, counts <= max_freq)
    # create gensim dictionary / BOW
    dct = Dictionary([tokens[ind]])
    dct.compactify()
    return dct


def create_bow(data, column, min_freq=0, max_freq=1, stopwords=None):
    """Returns a bag-of-word (BOW) object from a dataframe column of text using the gensim library.

    :param data: input dataframe
    :param column: column of the dataframe containing the text data
    :param min_freq: minimum word frequency below which tokens will be discarded from the BOW
    :param max_freq: maximum word frequency above which tokens will be discarded from the BOW
    :param stopwords: list of stopwords to exclude from the output
    :return: BOW (Dictionary gensim object)
    """
    text = " ".join(data[column])
    return create_bow_from_text(
        text, min_freq=min_freq, max_freq=max_freq, stopwords=stopwords
    )


def get_text(data, column, col_keywords=None, stopwords=None, unique=False):
    """Returns a dataframe column text content in normalized format wherever the content of the column matches
    the columns-keyword pair in the col_keyword variable.

    Example::
      get_text(df_soa.loc[df_soa.numero_sejour.isin(sej_exacts)], 'fnd_value',
               stopwords=stopwords, col_keywords=[('fnd_code', 'C1_MED_Ant_001')])

    :param data: input dataframe
    :param column: column containing the text data
    :param col_keywords: list of tuple (column, value) that has to match for the text to be selected
    :param stopwords: list of stopwords to exclude from the output
    :param unique: boolean flag (default False), if set to True constructs the new string with unique tokens only
    :return: extracted normalized string of text
    """
    text = ""
    if col_keywords is not None:
        ind = np.zeros(data.shape[0], dtype=np.bool)
        for c, w in col_keywords:
            ind = np.logical_or(data[c] == w, ind)
    else:
        ind = np.ones(data.shape[0], dtype=np.bool)
    text += " ".join(
        normalize_df_column(
            data.loc[ind], column=column, unique=unique, stopwords=stopwords
        )
    )
    if unique:
        return " ".join(np.unique(text.split()))
    return " ".join(text.split())


def text_to_bow_idx(data, column, bow):
    """Returns BOW indices corresponding to each words in the dataframe column.
    Ignores words that do not appear in the original BOW.
    Returns only one index per word, no matter how many times it appears on a given row.

    :param data: input dataframe
    :param column: text column where the indices will be computed (not necessarily normalized)
    :param bow: BOW gensim model
    :return: panda series containing the BOW indices corresponding to the original tokens
    """
    df = normalize_df_column(data, column)
    return df.apply(
        lambda x: [b[0] for b in bow.doc2bow(x.split(), return_missing=False)]
    )


def text2vec(model, finding_text, stopwords=None, agg="mean", weighted=False, N=None):
    """Returns the vector corresponding to a text, where the vector components are aggregated using the agg() method.

    Only acceptable aggregation methods : mean, std, min and max

    :param model: Word2Vec Gensim model
    :param finding_text: input text string
    :param stopwords: list of stopwords to exclude from the computation
    :param agg: aggregation function used to gather the result from all of the embedding vectors into a single one
    :param weighted: boolean flag (defaults to False), if True, weights the word vectors in inverse proportion to their count in the corpus (i.e. idf) (see https://arxiv.org/pdf/1607.00570.pdf)
    :param N: scaling factor for the weighted embeddings computation
    :return: the aggregated Word2Vec vector corresponding to the input text
    """
    if not isinstance(finding_text, str):
        return np.nan
    tokens = tokenize_text(finding_text, stopwords)
    if weighted:
        if N is None:
            N = np.max(
                [model.wv.get_vecattr(w, "count") for w in model.wv.key_to_index]
            )
        vecs = np.array(
            [
                model.wv[tok]
                * np.log((N + 1) / (model.wv.get_vecattr(tok, "count") + 1))
                for tok in tokens
                if tok in model.wv.key_to_index
            ]
        )
    else:
        vecs = np.array(
            [model.wv[tok] for tok in tokens if tok in model.wv.key_to_index]
        )
    if vecs.shape[0] == 0:
        return np.nan
    elif agg == "mean":
        return np.mean(vecs, axis=0)
    elif agg == "min":
        return np.min(vecs, axis=0)
    elif agg == "max":
        return np.max(vecs, axis=0)
    elif agg == "std":
        return np.std(vecs, axis=0)
    else:
        raise NameError(f"Unknown aggregation function: {agg}")


def col2vec(model, data, col, stopwords=None, agg="mean", weighted=False, suffix=""):
    """Returns the original dataframe with an additional column containing the aggregated Word2Vec representation of
    the text in the 'col' column.
    This function uses the text2vec() function to compute the Word2Vec representations.
    The name of the new column added to the input dataframe is constructed from the original 'col' name by appending
    a '_wv' suffix at the end.

    :param model: Word2Vec Gensim model
    :param data: input dataframe
    :param col: name of the column that contains the text data used to compute the embeddings
    :param stopwords: list of stopwords to exclude from the computation
    :param agg: aggregation function used to gather the result from all of the embedding vectors into a single one
    :param weighted: boolean flag (defaults to False), if True, weights the word vectors in inverse proportion to their count in the corpus (i.e. idf) (see https://arxiv.org/pdf/1607.00570.pdf)
    :param suffix: string suffix to add to the new dataframe column (in addition to the _wv default suffix)
    :return: input dataframe with an additional column containing the Word2Vec aggregated representation
    """
    sz = model.vector_size
    wvcol = f"{col}_wv" + suffix
    N = np.max([model.wv.get_vecattr(w, "count") for w in model.wv.key_to_index])
    df = data.loc[:, col].apply(
        lambda x: text2vec(model, x, stopwords, agg, weighted, N)
    )

    # reshape the dataframe to expand w2vec vectors and give each coordinates its single columns
    df = df.dropna()
    df = pd.DataFrame(df.values.tolist(), index=df.index)
    new_cols = [f"{wvcol}_{k:02d}" for k in range(sz)]
    df.columns = new_cols
    data = data.join(df)
    data.loc[:, new_cols] = data.loc[:, new_cols].fillna(
        0
    )  # replace missing information by null vectors
    return data
