import torch

def compute_mlds(logit_list, full_scale_target, num_classes, normalization_function='sigmoid'):

    # Form binary target and reshape for channels to be second dimension
    binary_full_scale_target = torch.nn.functional.one_hot(full_scale_target, num_classes=num_classes). \
        permute(0, -1, *tuple(list(range(len(full_scale_target.shape)))[1:])).float()

    # Select respective output normalization function
    if normalization_function == 'softmax':
        norm_funct = torch.nn.Softmax(dim=1)
    elif normalization_function == 'sigmoid':
        norm_funct = torch.nn.Sigmoid()
    else:
        raise NotImplementedError(
            "The normalization function {} has not been implemented for mlds.".format(normalization_function))

    criterion = torch.nn.BCELoss(reduction='none')
    multi_label_deep_supervision_loss = 0.0

    for current_logit in logit_list:
        # Down-scale target via max pooling
        max_pool = torch.nn.AdaptiveMaxPool2d(output_size=current_logit.shape[2:])
        current_target = torch.zeros_like(current_logit)
        for class_index in range(num_classes):
            current_target[:, class_index, :] = max_pool(binary_full_scale_target[:, class_index, :])

        # Sum up the loss for all intermediate-logits
        multi_label_deep_supervision_loss += criterion(norm_funct(current_logit), current_target).mean(
            tuple(list(range(len(current_logit.shape)))[1:])).sum()

    return multi_label_deep_supervision_loss
