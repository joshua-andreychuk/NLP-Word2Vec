# NLP-Word2Vec

# Understanding and Implementing Word2Vec

## Overview
In this assignment I re‑implemented from scratch in Python with NumPy the core Word2Vec skip‑gram model (both naive‑softmax and negative sampling variants) , then trained 10‑dimensional word vectors on the Stanford Sentiment Treebank. I also built an SGD optimizer and visualized a small set of learned embeddings in 2D.

## Key Features & Highlights
- **`sigmoid` implementation**: Vectorized sigmoid function.  
- **`naive_softmax_loss_and_gradient`**: Computes full‑vocabulary softmax loss and its gradients w.r.t. the center vector and all outside vectors.  
- **`neg_sampling_loss_and_gradient`**: Efficient negative‑sampling loss with K=10 negatives per positive, plus correct gradient accumulation for repeated samples.  
- **`skipgram` model**: Orchestrates loss/gradient calls over all context words for a given center word.  
- **`sgd` optimizer**: General‑purpose stochastic gradient descent with learning‑rate annealing, optional parameter checkpointing, and post‑processing hooks.  
- **`run.py` training & visualization**:  
  - Initializes a stacked (input + output) vector matrix;  
  - Trains for 40 000 iterations with batchsize=50 and step size=0.3;  
  - Outputs running loss (“sanity check: cost at convergence ≲ 10”);  
  - Saves `sample_vectors_(soln).json`;  
  - Produces `word_vectors_(soln).png` via PCA and Matplotlib.

## Results

| Item                            | Value / Path                       |
|---------------------------------|------------------------------------|
| Final (running‐average) loss    | 9.8                             |
| Sample vectors dump             | `sample_vectors_(soln).json`       |
| 2D embedding visualization      | `word_vectors_(soln).png`          |

## Insights
- **Negative Sampling vs. Full Softmax**  
  Negative sampling drastically reduces per‑step cost by only updating K + 1 vectors rather than the entire vocabulary, yielding faster convergence without sacrificing embedding quality.
- **Embedding Semantics**  
  The learned vectors cluster semantically and sentimentally: positive adjectives (e.g. _great_, _wonderful_) form one region, negatives (_bad_, _boring_) another, and gender terms (_king_, _queen_) align along coherent axes.

## Project Structure
- **submission.py**  
  All core implementations:  
  - `sigmoid`  
  - `naive_softmax_loss_and_gradient`  
  - `neg_sampling_loss_and_gradient`  
  - `skipgram`  
  - `word2vec_sgd_wrapper`  
  - `sgd` optimizer  
- **run.py**  
  Training script, sample‑context loader, PCA projection and plotting.  
- **sample_vectors_(soln).json**  
  10‑dimensional vectors for a fixed set of words, used for autograder evaluation.  
- **word_vectors_(soln).png**  
  Scatterplot of the above sample words in 2D.  

## Assignment Requirements Met
1. **Correctness**: Passed gradient checks for both loss functions; final training loss converges below the target threshold.  
2. **Completeness**: Implemented all functions specified in the prompt and hooked them together in `run.py`.  
