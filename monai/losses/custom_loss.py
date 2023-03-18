import torch
import torch.nn as nn
from monai.losses.unified_loss import AsymmetricUnifiedFocalLoss


class CustomLoss(nn.Module):
    """Weighted Asymmetric Unified Focal loss and MSE on Volume ratios
    """
    def __init__(self, weightAUF=0.75, weightBVTV=1., weight=0.5, delta=0.6, gamma=0.2, epsilon=1e-07):
        super(CustomLoss, self).__init__()
        self.weightAUF = weightAUF      # weights
        self.weightBVTV = weightBVTV    # weights
        self.weight = weight            # auf
        self.delta = delta              # auf
        self.gamma = gamma              # auf
        self.epsilon = epsilon          # bvtv_mse

    def forward(self, y_pred, y_true):
        # Obtain Asymmetric Unified Focal loss
        auf_loss = AsymmetricUnifiedFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)

        # Obtain BVTV ration mse
        vols_pred = torch.sum(y_pred, dim=[2, 3, 4])
        bvtv_pred = (vols_pred[:, 1] + self.epsilon) / (vols_pred[:, 0] + self.epsilon)
        vols_true = torch.sum(y_true, dim=[2, 3, 4])
        bvtv_true = (vols_true[:, 1] + self.epsilon) / (vols_true[:, 0] + self.epsilon)
        bvtv_mse = nn.MSELoss()(bvtv_pred, bvtv_true)

        # Return weighted sum of AUF loss and BVTV mse
        return self.weightAUF * auf_loss + self.weightBVTV * bvtv_mse


if __name__ == "__main__":
    shapes = (100, 2, 10, 10, 10)
    y_true = torch.randint(low=0, high=2, size=shapes)
    y_pred = 1 - y_true # torch.rand(shapes)
    print(CustomLoss()(y_true, y_pred))
