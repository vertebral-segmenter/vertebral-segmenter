import torch
import torch.nn as nn
from monai.losses.unified_loss import AsymmetricUnifiedFocalLoss
from monai.networks import one_hot


class CustomLoss(nn.Module):
    """Weighted Asymmetric Unified Focal loss and MSE on Volume ratios
    """
    def __init__(self, to_onehot_y=False, weightAUF=2., weightBVTV=1., weight=0.5, delta=0.6, gamma=0.8, epsilon=1e-07):
        super(CustomLoss, self).__init__()
        self.weightAUF = weightAUF      # weights
        self.weightBVTV = weightBVTV    # weights
        self.weight = weight            # auf
        self.delta = delta              # auf
        self.gamma = gamma              # auf
        self.epsilon = epsilon          # bvtv_mse
        self.to_onehot_y = to_onehot_y

    def forward(self, y_pred, y_true):
        # Obtain Asymmetric Unified Focal loss
        if self.to_onehot_y:
            y_true = one_hot(y_true, num_classes=y_pred.shape[1])
        auf_loss = AsymmetricUnifiedFocalLoss(weight=self.weight, delta=self.delta, gamma=self.gamma)(y_pred, y_true)
        # Obtain BVTV ration mse
        vols_pred = torch.sum(y_pred, dim=[2, 3, 4])
        bvtv_pred = (vols_pred[:, 1]) / (vols_pred[:, 0] + vols_pred[:, 1])
        vols_true = torch.sum(y_true, dim=[2, 3, 4])
        bvtv_true = (vols_true[:, 1]) / (vols_true[:, 0] + vols_true[:, 1])
        bvtv_mse = nn.MSELoss()(bvtv_pred, bvtv_true)

        # Return weighted sum of AUF loss and BVTV mse
        return self.weightAUF * auf_loss + self.weightBVTV * bvtv_mse


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x_lst = []
    y_lst = []
    for i in range(0, 96**3, 1000):
        tr = int(96**3 / 2)
        y_true = torch.cat((torch.ones(tr), torch.zeros(96**3-tr))).reshape(1, 1, 96, 96, 96)
        fr = i
        y_pred = torch.cat((torch.ones(fr), torch.zeros(96**3-fr))).reshape(1, 1, 96, 96, 96)
        y_pred = one_hot(y_pred, num_classes=2)
        x_lst.append(i)
        y_lst.append(CustomLoss(to_onehot_y=True)(y_pred, y_true))
    plt.plot(x_lst, y_lst)
    plt.show()

