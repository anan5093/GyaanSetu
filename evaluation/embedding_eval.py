from sklearn.decomposition import PCA

class EmbeddingEvaluator:

    def __init__(self, embedder):
        self.embedder = embedder

    def evaluate(self, words):

        vectors = self.embedder.embed(words)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)

        return [
            {
                "word": words[i],
                "x": float(reduced[i][0]),
                "y": float(reduced[i][1])
            }
            for i in range(len(words))
        ]
