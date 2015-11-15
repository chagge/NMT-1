# NMT

This repo contains code I've been working on for neural machine translation using information from dependency parses. It's written in python using the Theano deep learning API (although with the release of Google's TensorFlow, I'll probably switch it to that once I have time). At the moment, only the code for building GloVe vectors (Pennington, Socher, Manning 2014) still works--everything else is deprecated or incomplete.

Since it's primarily designed for my own use, the documentation is essentially nonexistant. Here's basically what's in each file:
* glove.py: builds GloVe vectors from a corpus of files containing tuples of (word, POS tag?, dep label?)
* nmt.py: Sequence-to-sequence model. Incomplete.
* RNN.py, RNN1.py, ..., RNN4.py: Recurrent NN module, builds various kinds of recurrent models. Deprecated.
* UNKRNN.py: SUPER deprecated.
* gridSearch.sh: bash script for training multiple models in sequence.
* matwizard.py: module for initializing matrices very particularly, ensuring that the distribution of input/output values at each layer maintains certain desirable properties that facilitate early-stage learning.
* funx.py: module with various auxiliary functions, including a bunch of different squashing functions and cost functions.
