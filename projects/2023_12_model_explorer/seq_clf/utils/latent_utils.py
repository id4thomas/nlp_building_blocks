import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def reduce_dim_with_tsne(x):
	x_embedded = TSNE(
	n_components=2, 
	learning_rate='auto',
	init='random',
	perplexity=10).fit_transform(x)
	return x_embedded