import numpy as np

from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder


def calculate_projection(v, z=1):

    no_features = v.shape[0]
    var_u = np.sort(v)[::-1]
    cssv = np.cumsum(var_u) - z
    var_ind = np.arange(no_features) + 1
    ind = var_u - cssv / var_ind > 0
    var_r = var_ind[ind][-1]
    theta = cssv[ind][-1] / float(var_r)
    var_w = np.maximum(v - theta, 0)
    return var_w


class MulticlassSVM():

    def __init__(self, C=1, max_iter=50, tol=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol,
        self.random_state = random_state
        self.verbose = verbose

    def calculate_vio(self, g, y, i):
        # Optimality violation for the ith sample.
        small = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dual_coeffi[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_coeffi[k, i] >= 0:
                continue

            small = min(small, g[k])

        return g.max() - small

    def calculate_prediction(self, X):
        dec = np.dot(X, self.coeffi.T)
        pre = dec.argmax(axis=1)
        return self.label_encoder.inverse_transform(pre)

    def calculate_partial_gradient(self, X, y, i):
        # Partial gradient for the ith sample.
        partial_grad = np.dot(X[i], self.coeffi.T) + 1
        partial_grad[y[i]] -= 1
        return partial_grad

    def calculate_subproblem(self, g, y, norms, i):
        # Prepare inputs to the projection.
        tempC = np.zeros(g.shape[0])
        tempC[y[i]] = self.C
        beta_hat = norms[i] * (tempC - self.dual_coeffi[:, i]) + g / norms[i]
        z = self.C * norms[i]

        # Compute projection onto the simplex.
        beta = calculate_projection(beta_hat, z)

        return tempC - self.dual_coeffi[:, i] - beta / norms[i]

    def svm_findOtherParameters(self, confusion_mat):

        list_diagonal = np.zeros(confusion_mat.shape[0])
        list_row_sum = np.zeros(confusion_mat.shape[0])
        list_column_sum=np.zeros(confusion_mat.shape[1])

        precision_value = []
        recall_value = []
        f_measure_value = []

        total = np.sum(confusion_mat)
        confuse_diagonal = 0

        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                list_row_sum[i] += confusion_mat[i][j]
                list_column_sum[i] += confusion_mat[j][i]
                if(i==j):
                    list_diagonal[i] = confusion_mat[i][j]
                    confuse_diagonal +=  confusion_mat[i][j]
        # print "Accuracy", float(confuse_diagonal)/total
        accuracy = float(confuse_diagonal)/total

        for index in range(len(list_row_sum)):
            if list_row_sum[index]==0:
                precision_value.append(0.0)
            else:
                precision_value.append((float)(list_diagonal[index]) / list_row_sum[index])

            if list_column_sum[index]==0:
                recall_value.append(0)
            else:
                recall_value.append((float)(list_diagonal[index]) / list_column_sum[index])

            if precision_value[index]==0 or recall_value[index]==0:
                f_measure_value.append(0)
            else:
                f_measure_value.append((float) (2 * precision_value[index] * recall_value[index]) / (precision_value[index] + recall_value[index]))

        return accuracy, precision_value, recall_value, f_measure_value

    def fit(self, X, y):
        no_samples, no_features = X.shape

        # labels - Normalize
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        # primal and dual coefficients.
        no_classes = len(self.label_encoder.classes_)
        self.dual_coeffi = np.zeros((no_classes, no_samples), dtype=np.float64)
        self.coeffi = np.zeros((no_classes, no_features))

        # Pre-compute norms.
        norms = np.sqrt(np.sum(X ** 2, axis=1))

        # Shuffle  indices.
        ran_state = check_random_state(self.random_state)
        index = np.arange(no_samples)
        ran_state.shuffle(index)

        vio_init = None
        for it in range(self.max_iter):
            vio_sum = 0

            for ii in range(no_samples):
                i = index[ii]

                # All-zero samples can be safely ignored.
                if norms[i] == 0:
                    continue

                g = self.calculate_partial_gradient(X, y, i)
                v = self.calculate_vio(g, y, i)
                vio_sum += v

                if v < 1e-12:
                    continue

                # Solve subproblem for the ith sample.
                delta = self.calculate_subproblem(g, y, norms, i)

                # Update primal and dual coefficients.
                self.coeffi += (delta * X[i][:, np.newaxis]).T
                self.dual_coeffi[:, i] += delta

        return self