import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

color_palette1 = list(mcolors.TABLEAU_COLORS)
color_palette2 = list(mcolors.XKCD_COLORS)
np.random.shuffle(color_palette2)
color_palette = color_palette1 + color_palette2

embeddings, labels = read_ms1m(ms1m_dir)
df = pd.read_csv(Path(output_dir) / 'MS1M_centroid_0.4.csv')
preds = df['pred_label'].values

embeddings = embeddings[labels < 10]
preds = preds[labels < 10]
labels = labels[labels < 10]

pred_classes, counts= np.unique(preds, return_counts=True)
pred_classes = pred_classes[np.argsort(counts)[::-1]]
new_preds = preds.copy()
for i, c in enumerate(pred_classes):
    new_preds[preds == c] = i 
preds = new_preds

points = TSNE(
    n_components=2,
    perplexity=10.0,
    learning_rate='auto',
    n_iter=1000,
    metric='cosine',
    init='pca',
    verbose=1,
    square_distances=True
).fit_transform(embeddings)

colors = [color_palette[l] for l in preds]
plt.scatter(points[:, 0], points[:, 1], marker='.', c=colors, s=25)
plt.axis('off')
plt.show()