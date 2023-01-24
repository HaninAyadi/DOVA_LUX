"""
Definitions of visualization methods
"""

from wordcloud import WordCloud
from yellowbrick.text import TSNEVisualizer
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["axes.titlesize"] = 15


def create_wordcloud(data, title=None):
    wordcloud = WordCloud(width=500, height=500,
                          background_color='white',
                          min_font_size=15).generate(" ".join(data))

    plt.figure(figsize=(5, 5), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=20)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def tsne_viz(X, labels, colors=None, colormap='gist_ncar'):
    if colors == None:
        tsne = TSNEVisualizer(colormap=colormap)
    else:
        tsne = TSNEVisualizer(colors=colors)

    tsne.fit(X, labels)
    tsne.show()
    tsne.set_title('t-SNE Visualization of Grouped Clusters')
