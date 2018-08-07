# ULMFiT_reproduce

To reproduce the results of ULMFiT

# Author's results:
|     | IMDb         | TREC-6       | AG           | Yelp-bi      | Yelp-full    | DBpedia      |
|-----| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|`avg`| 95.4 (4.6)   | 96.4 (3.6)   | 94.99 (5.01) | 97.84 (2.16) | 70.02 (29.98)| 99.2 (0.80)  |

# My results:
    - ....: tuning in progress
    - ??.?: to be tuned
|     | IMDb         | TREC-6       | AG           | Yelp-bi      | Yelp-full    | DBpedia      |
|-----| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|`fwd`| 94.3  (5.7)  | ??.?  (?.?)  | 94.2  (5.8)  | ....  (...)  | ??.?  (?.?)  | ....  (...)  |
|`bwd`| ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  |
|`avg`| ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  |

Dataset: imdb, trec-6, ag, yelp-bi, yelp-full, dbpedia

1. Convert each dataset into csv file: tran.csv & val.csv
    - classification path: has info to create sentiment analysis model (label & data)
    - language model path: has info to create language model (no labels)

2. Tokenize train.csv & val.csv
      - tok_trn.npy & tok_val.npy: 1-d array of tokenized text
        (tok_trn[0]: "\n xbox xfld 1 having watched this for the third time , it kills ...")
      - lbl_trn.npy & lbl_val.npy: 1-d array of labels(int)
        (lbl_trn[0]: 1)

3. Numericalize the tokens
      - trn_ids.npy & val_ids.npy: 1-d array of ids that map to tokens
        (trn_ids[0]: '40 41 42 39 279 320 13 22 2 835 74 4 10 ...')
      - itos.pkl: mapping between token id & token text
  
4. We have pretrained LM on Wikitext-103
      - Train imdb LM which starts with the weights of the pretrained LM on Wikitext-103
      - imdb LM must have the same embedding size, # hidden layers as pretrained LM's
      - Map imdb vocabulary to wikitext-103 vocabulary
  
  
