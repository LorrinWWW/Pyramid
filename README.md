# README

Code associated with the paper [**Pyramid: A Layered Model for Nested Named Entity Recognition**](https://www.aclweb.org/anthology/2020.acl-main.525/) at ACL 2020.

## Citation

```bibtex
@inproceedings{jue2020pyramid,
  title={Pyramid: A Layered Model for Nested Named Entity Recognition},
  author={Wang, Jue and Shou, Lidan and Chen, Ke and Chen, Gang},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={5918--5928},
  year={2020}
}
```

## Runing Experiments

1. Preprocess the corpora

   For ACE and GENIA, we follow the script from https://github.com/yahshibu/nested-ner-tacl2020-transformers to preprocess the corpora. For NNE, we use the proprocessing script from https://github.com/nickyringland/nested_named_entities.

    Each dataset is further unified in json and placed in "./datasets/unified/" as "train.{dataset}.json", "valid.{dataset}.json", and "test.{dataset}.json" three files.

   Each json file consists of a list items, and each of item looks like:

   ```json
   {
     "tokens": ["token0", "token1", "token2"],
     "entities": [
       {
         "entity_type": "PER", 
         "span": [0, 1],
       },
       {
         "entity_type": "ORG", 
         "span": [2, 3],
       },
     ]
   }
   ```

2. Generate embeddings

   Then we prepare the pretrained word embeddings, such as GloVe. Available at https://nlp.stanford.edu/projects/glove/.

   Each line of this file represents a token or word. Here is an example with a vector of length 5:

   ```
   word 0.002 1.9999 4.323 4.1231 -1.2323
   ```

   (Optional) It is also recommended to use language model based contextualized embeddings, such as BERT. Check "./run/gen_XXX_emb.py" to generate them.

3. Start training
   
   Run the following cmd to start the training, e.g., on ACE05.

   ```bash
   $ python train_ner.py \
           --batch_size 32 \
           --evaluate_interval 500 \
           --dataset ACE05 \
           --pretrained_wv ../wv/PATH_TO_WV_FILE  \
           --max_epoches 500 \
           --model_class PyramidNestNER  \
           --model_write_ckpt ./PATH_TO_CKPT_TO_WRITE \
           --optimizer sgd \
           --lr 0.01 \
           --tag_form iob2  \
           --cased 0 \
           --token_emb_dim 100 \
           --char_emb_dim 30 \
           --char_encoder lstm \
           --lm_emb_dim 0 \
           --lm_emb_path ../wv/PATH_TO_LM_EMB_PICKLE_OBJECT \
           --tag_vocab_size 100 \
           --vocab_size 20000 \
           --dropout 0.4 \
           --max_depth 16
   ```
   
   Log samples are placed in "./logs/"
   

## Arguments

### --batch_size

The batch size to use.

### --evaluate_interval

The evaluation interval, which means evaluate the model for every {evaluate_interval} training steps.

### --dataset

The name of dataset to be used. The dataset should be unified in json and placed in "./datasets/unified/", ias"train.{dataset}.json", "valid.{dataset}.json", and "test.{dataset}.json" three files.

Each json file consists of a list items, and each of item is as follows:

```json
{
  "tokens": ["token0", "token1", "token2"],
  "entities": [
    {
      "entity_type": "PER", 
      "span": [0, 1],
    },
    {
      "entity_type": "ORG", 
      "span": [2, 3],
    },
  ]
}
```

### --pretrained_wv

The pretrained word vectors file, such as GloVe.

Each line of this file represents a token or word. Here is an example with a vector of length 5:

```
word 0.002 1.9999 4.323 4.1231 -1.2323
```

### --max_epoches

max_epoches

### --model_class

model_class, should be **PyramidNestNER** or **BiPyramidNestNER**

### --model_write_ckpt

Path of model_write_ckpt. None if you don't want to save checkpoints.

### --optimizer

Optimizer to use. "adam" or "sgd".

### --lr

Learning rate. E.g. 1e-2.

### --tag_form

tag_form. Currently only support IOB2.

### --cased

Whether cased for word embeddings (0 or 1). Note for char embs, it is always cased.

### --token_emb_dim

Word embedding dimension. This should be in line with "pretrained_wv" file.

### --char_emb_dim

Character embedding dimension. 0 to disable it.

30 works fine.

### --char_encoder

Use "lstm" or "cnn" char encoder. 

### --lm_emb_dim

Language model based embedding dimension. 0 to disable it.

### --lm_emb_path

Language model embeddings. "lm_emb_path" is required if "lm_emb_dim" > 0.

which is a pickle file, representing a dictionary object, mapping a tuple of tokens to a numpy matrix:

```json
{
  (t0_0,t0_1,t0_2,...,t0_23): np.array([24, 1024]),
  (t1_0,t1_1,t1_2,...,t1_16): np.array([17, 1024]),
  ...
}
```

check "./run/gen_XXX_emb.py" to know how to generate the language model embeddings.

### --tag_vocab_size

Maximum of tag vocab size. A value bigger than the possible number of IOB2 tags.

### --vocab_size

Maximum of token vocab size.

### --dropout

dropout rate

### --max_depth

Max height for the Pyramid.

Bigger for better support for longer nested entities;

Smaller for quicker training/inference speed.

## Model

The model is defined in "./models/pyramid_nest_ner.py".

Feel free to modify and test it.
