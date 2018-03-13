import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.metrics import roc_curve, auc


def parse(f_name):
    train = []

    with open(f_name, 'rb') as f:
        b = f.read(1)

        while b != b"":
            data = []

            for i in range(400):
                data.append(float(ord(b)) / 255)
                b = f.read(1)
                if b == b"":
                    break

            train.append(data)
    return train


def check(dataset):
    for row in dataset:
        if len(row) != 400:
            print('Len(dataset.row) ERROR must be 400')


# Plot an ROC. pred - the predictions, y - the expected output.
def mult_plot_roc(pred, y, classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    cmap = get_cmap(name='hsv', lut=n_classes)
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=cmap(i), lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
