"""
Definitions of basic visualization methods

Author: Hanin Ayadi
Creation date: 15/06/2022

Inspired by

"""
from wordcloud import WordCloud
from yellowbrick.text import TSNEVisualizer
import matplotlib.pyplot as plt


def create_wordcloud(data, title=None):
    wordcloud = WordCloud(width=500, height=500,
                          background_color='white',
                          min_font_size=15
                          ).generate(" ".join(data))
    plt.figure(figsize=(5, 5), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(title, fontsize=20)
    plt.show()


def tsne_viz(X, labels, colormap='gist_ncar'):
    tsne = TSNEVisualizer(colormap)
    tsne.fit(X, labels)
    tsne.show()
