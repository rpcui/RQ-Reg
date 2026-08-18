import numpy as np
from kmeans import KMeans

class ResCluster:
    def __init__(self, method, n_layers, codebook_size, dim, lambd):
        if type(method) == str:
            method = [method] * n_layers
        self.method = method
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.dim = dim
        self.lambd = lambd
        self.epsilon = 5e-3
        self.codebooks = [None for _ in range(n_layers)]
        self.num_samples = [None for _ in range(n_layers)]
        self.models = []
        for n in range(n_layers):
            if method[n] == "kmeans":
                self.models.append(KMeans(n_clusters=self.codebook_size[n], verbose=True))
            else:
                raise ValueError("Unsupported clustering method: {}".format(method[n]))

    def _l1_distance(self, x, y):
        return np.sum(np.abs(x[:, None, :] - y[None, :, :]), axis=2)

    def _quant_distance(self, x, y):
        d = x[:, None, :] - y[None, :, :]
        d[d < 0] = np.inf
        return np.sum(d, axis=2)

    def train(self, inputs):
        x = inputs
        sample_mask = np.ones((x.shape[0],), dtype=int)
        quant_codes = np.zeros((x.shape[0], self.n_layers))
        codebooks = []
        for l in range(self.n_layers):
            cluster_model = self.models[l]
            labels = cluster_model.fit_predict(x, sample_mask)

            cluster_num_samples = [sum(sample_mask)] + [sum((labels == i) * sample_mask) for i in range(self.codebook_size[l])]
            self.num_samples[l] = cluster_num_samples
            codebooks.append(np.array(sorted(cluster_model.centers_)))

            quant = cluster_model.centers_[labels]
            quant_codes[:, [l]] = quant * sample_mask[:, None]
            residuals = (x - quant) * sample_mask[:, None]

            sample_mask = sample_mask * np.all(np.abs(residuals) > self.epsilon, axis=1).astype(int)
            x = residuals

        self.quant_codes = quant_codes
        self.codebooks = codebooks

    def encode(self, inputs):
        num = inputs.shape[0]
        codes = np.zeros((num, self.n_layers), dtype=int)
        quant_codes = np.zeros((num, self.n_layers))
        num_samples = np.zeros((self.n_layers, max(self.codebook_size) + 1))
        eos = np.zeros((num,), dtype=int)
        x = np.reshape(inputs, (num, self.dim))
        for l in range(self.n_layers):
            distance = self._l1_distance(x, self.codebooks[l]) * (1 - eos[:, None])
            codes[:, l] = np.where(eos, -1, np.argmin(distance, axis=1))
            quant_codes[:, [l]] = np.where(eos[:, None], 0.0, self.codebooks[l][codes[:, l]])

            num_samples[l, 0] = sum(1 - eos)
            num_samples[l, 1: self.codebook_size[l] + 1] = [sum(codes[:, l] == i) for i in range(self.codebook_size[l])]

            x = (x - quant_codes[:, [l]]) * (1 - eos[:, None])
            eos = np.all(np.abs(x) < self.epsilon, axis=1).astype(int)

        return codes, quant_codes, num_samples
