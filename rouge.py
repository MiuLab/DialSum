"""ROUGE metric implementation.

Copy from tf_seq2seq/seq2seq/metrics/rouge.py.
This is a modified and slightly extended verison of
https://github.com/miso-belica/sumy/blob/dev/sumy/evaluation/rouge.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import numpy as np

#pylint: disable=C0103

def _get_ngrams(n, text):
  """Calcualtes n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set

def _get_word_ngrams(n, words):
  """Calculates word n-grams for multiple sentences.
  """
  assert len(words) > 0
  assert n > 0

  return _get_ngrams(n, words)


def _len_lcs(x, y):
  """
  Returns the length of the Longest Common Subsequence between sequences x
  and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: sequence of words
    y: sequence of words

  Returns
    integer: Length of LCS between x and y
  """
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]

def _lcs(x, y):
  """
  Computes the length of the longest common subsequence (lcs) between two
  strings. The implementation below uses a DP programming algorithm and runs
  in O(nm) time where n = len(x) and m = len(y).
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: collection of words
    y: collection of words

  Returns:
    Table of dictionary of coord and len lcs
  """
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table

def rouge_n(evaluated_sentences, reference_sentences, n=2):
  """
  Computes ROUGE-N of two text collections of sentences.
  Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
  papers/rouge-working-note-v1.3.1.pdf

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
    n: Size of ngram.  Defaults to 2.

  Returns:
    A tuple (f1, precision, recall) for ROUGE-N

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  #remove paddings
  try:
    end = evaluated_sentences.index(0)
    evaluated_sentences = evaluated_sentences[:end]
  except ValueError:
    pass
  try:
    end = reference_sentences.index(0)
    reference_sentences = reference_sentences[:end]
  except ValueError:
    pass

  precision = 0.0
  recall = 0.0
  f1_score = 0.0

  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    return f1_score, precision, recall

  evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
  reference_ngrams = _get_word_ngrams(n, reference_sentences)
  reference_count = len(reference_ngrams)
  evaluated_count = len(evaluated_ngrams)

  if (evaluated_count == 0) or (reference_count == 0):
    return f1_score, precision, recall

  # Gets the overlapping ngrams between evaluated and reference
  overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
  overlapping_count = len(overlapping_ngrams)

  precision = overlapping_count / evaluated_count
  recall = overlapping_count / reference_count
  if(precision == 0) and (recall == 0):
    f1_score = 0.0
  else:
    f1_score = 2.0 * ((precision * recall) / (precision + recall))

  # return overlapping_count / reference_count
  return f1_score, precision, recall

def _f_p_r_lcs(llcs, m, n):
  """
  Computes the LCS-based F-measure score
  Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Args:
    llcs: Length of LCS
    m: number of words in reference summary
    n: number of words in candidate summary

  Returns:
    Float. LCS-based F-measure score
  """
  r_lcs = llcs / m
  p_lcs = llcs / n
  beta = p_lcs / (r_lcs + 1e-12)
  num = (1 + (beta**2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta**2) * p_lcs)
  f_lcs = num / (denom + 1e-12)
  return f_lcs, p_lcs, r_lcs

def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (sentence level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set

  Returns:
    F_lcs, P_lcs, R_lcs

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  #remove paddings
  try:
    end = evaluated_sentences.index(0)
    evaluated_sentences = evaluated_sentences[:end]
  except ValueError:
    pass
  try:
    end = reference_sentences.index(0)
    reference_sentences = reference_sentences[:end]
  except ValueError:
    pass

  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    return 0.0,0.0,0.0
  m = len(reference_sentences)
  n = len(evaluated_sentences)
  lcs = _len_lcs(evaluated_sentences, reference_sentences)
  return _f_p_r_lcs(lcs, m, n)

def rouge(hypotheses, references):
  """Calculates average rouge scores for a list of hypotheses and
  references"""

  # Filter out hyps that are of 0 length
  # hyps_and_refs = zip(hypotheses, references)
  # hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
  # hypotheses, references = zip(*hyps_and_refs)

  hyp = hypotheses
  ref = references
  # Calculate ROUGE-1 F1, precision, recall scores
  rouge_1 = [ rouge_n(hyp, ref, 1) ]
  rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

  # Calculate ROUGE-2 F1, precision, recall scores
  rouge_2 = [ rouge_n(hyp, ref, 2) ]
  rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

  # Calculate ROUGE-3 F1, precision, recall scores
  rouge_3 = [ rouge_n(hyp, ref, 3) ]
  rouge_3_f, rouge_3_p, rouge_3_r = map(np.mean, zip(*rouge_3))

  # Calculate ROUGE-L F1, precision, recall scores
  rouge_l = [ rouge_l_sentence_level(hyp, ref) ]
  rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))

  return {
      "rouge_1/f_score": rouge_1_f,
      "rouge_1/r_score": rouge_1_r,
      "rouge_1/p_score": rouge_1_p,
      "rouge_2/f_score": rouge_2_f,
      "rouge_2/r_score": rouge_2_r,
      "rouge_2/p_score": rouge_2_p,
      "rouge_3/f_score": rouge_3_f,
      "rouge_3/r_score": rouge_3_r,
      "rouge_3/p_score": rouge_3_p,
      "rouge_l/f_score": rouge_l_f,
      "rouge_l/r_score": rouge_l_r,
      "rouge_l/p_score": rouge_l_p,
  }
