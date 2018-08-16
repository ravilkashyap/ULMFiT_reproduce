# ULMFiT_reproduce

To reproduce the results of ULMFiT & customize fastai framework

# Perplexities on Wikitext-103 validation dataset 
- All LM trained on Wikitext-103 training set
1. Pretrained LM (download): 24.375
    - vocab_size: 238462
    - ?? epoches
2. LM I trained with Adam: 13.671
    - vocab_size: 30002
    - 12 epoches
3. LM I trained with SGD: 361.78
    - vocab_size: 30002
    - Very slow convergence 
    - 24 epoches

# Author's results:
|       | IMDb         | TREC-6       | AG           | Yelp-bi      | Yelp-full    | DBpedia      |
|-------| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|`bidir`| 95.4 (4.6)   | 96.4 (3.6)   | 94.99 (5.01) | 97.84 (2.16) | 70.02 (29.98)| 99.2 (0.80)  |

# My results:
    - Fine-tuned with pretrained LM on wikitext-103
    - bidir accuracy: avg of bwd & fwd (requires implementation)
|       | IMDb         | TREC-6       | AG           | Yelp-bi      | Yelp-full    | DBpedia      |
|-------| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| `fwd` | 94.35 (5.65) | 96.8  (3.2)  | 94.2  (5.8)  | 97.61 (2.39) | ??.?  (?.?)  | 99.02 (1.08) |
|`bidir`| ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  | ??.?  (?.?)  |

    - Fine-tuned with (i trained) LM on wikitext-103
|       | IMDb         | TREC-6 (6i)  | AG           | Yelp-bi      | Yelp-full    | DBpedia      |
|-------| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| `fwd` | --.-- (-.--) | 97.0  (3.0)  | ??.?  (?.?)  | ??.?? (?.??) | ??.?  (?.?)  | ??.?? (?.??) |

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
  
4. If we have pretrained LM on Wikitext-103
      - Train imdb LM which starts with the weights of the pretrained LM on Wikitext-103
      - imdb LM must have the same embedding size, # hidden layers as pretrained LM's
      - Map imdb vocabulary to wikitext-103 vocabulary
  
  
