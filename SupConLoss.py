
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)


        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        """all_zero = (mask == 0).all()
        print("Is mask all zeros:", all_zero)"""
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        """# 检查是否存在无穷大值（inf）
        has_inf = torch.isinf(exp_logits).any()
        print("Exp logits has inf values:", has_inf)

        # 检查是否存在非数值值（nan）
        has_nan = torch.isnan(exp_logits).any()
        print("Exp logits has nan values:", has_nan)"""

        """sum_exp_logits = exp_logits.sum(1, keepdim=True)
        if torch.any(sum_exp_logits <= 1e-20):  # 1e-20 是一个非常小的阈值
            print("下溢发生")"""

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+1e-12)

        """# 检查是否存在非数值值（nan）
        has_nan = torch.isnan(log_prob).any()
        print("Log prob has nan values:", has_nan)

        # 检查是否存在非数值值（inf）
        has_inf = torch.isinf(log_prob).any()
        print("Log prob has inf values:", has_inf)"""


        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]


        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs <= 1e-6, torch.tensor(1), mask_pos_pairs)
        """print("mask_pos_pairs:", mask_pos_pairs)
        # 检查 mask_pos_pairs 中是否存在0
        contains_zero = (mask_pos_pairs == 0).any().item()
        print("Does mask_pos_pairs contain zeros:", contains_zero)"""

        """if (mask_pos_pairs < 1e-6).any():
            print("mask_pos_pairs contains very small values which can cause NaN when dividing")
        if (log_prob <= 0).any():
            print("log_prob contains non-positive values")"""

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_pos_pairs + 1e-10)

        """# 检查 mean_log_prob_pos 是否包含任何 inf 值
        contains_inf = torch.isinf(mean_log_prob_pos).any().item()
        if contains_inf:
            print("mean_log_prob_pos contains Inf values")

        # 检查 mean_log_prob_pos 是否包含任何 nan 值
        contains_nan = torch.isnan(mean_log_prob_pos).any().item()
        if contains_nan:
            print("mean_log_prob_pos contains NaN values")"""


        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss