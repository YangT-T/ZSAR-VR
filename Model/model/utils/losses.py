import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter

'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
# def contrastive_loss(m_skel, m_txt, m_img, tmp, target, reduction='mean'):
#     # print("================")
    
#     logits_per_skel_txt = tmp * (m_skel @ m_txt.t())
#     logits_per_txt_skel = tmp * (m_txt @ m_skel.t())
#     loss = (F.cross_entropy(logits_per_skel_txt, target, reduction=reduction) + \
#             F.cross_entropy(logits_per_txt_skel, target, reduction=reduction))
#     loss_len = 1
#     if m_img != None:
#         logits_per_skel_img = tmp * (m_skel @ m_img.t())
#         logits_per_img_skel = tmp * (m_img @ m_skel.t())
#         loss += (F.cross_entropy(logits_per_skel_img, target, reduction=reduction) + \
#                 F.cross_entropy(logits_per_img_skel, target, reduction=reduction))
#         loss_len += 1
    
#     # loss = loss / loss_len
#     return loss, logits_per_skel_txt
    
# from collections import defaultdict, Counter
# import torch.nn.functional as F

# def accuracy(input, target, class_emb, is_test=False, count_by_batch=False, return_confusion=False):
#     # input: (bs, D)
#     # class_emb: (num_classes, D)
#     expand_input = input.unsqueeze(1).expand([input.shape[0], class_emb.shape[0]] + list(input.shape[1:]))
#     expand_class_emb = class_emb.unsqueeze(0).expand([input.shape[0], class_emb.shape[0]] + list(class_emb.shape[1:]))
    
#     final_scores = torch.sum(expand_input * expand_class_emb, axis=-1)
#     while len(final_scores.shape) > 2:
#         final_scores = final_scores.mean(-1)
        
#     temperature = 0.05  # 可调参数，推荐范围 0.05~0.2
#     final_scores = final_scores / temperature


#     final_scores = F.log_softmax(final_scores, 1)
#     preds = torch.argmax(final_scores, axis=1)  # (bs,)
#     # preds = torch.randint(low=0, high=5, size=(final_scores.shape[0],), device=final_scores.device)

#     confusion_stats = defaultdict(Counter)  # {true_label: Counter({pred_label: count})}
    
#     if target is not None:
#         acc_out = (preds == target)
#         acc = torch.sum(acc_out).item() / len(acc_out)

#         # === NEW: collect class-wise prediction distribution ===
#         if return_confusion:
#             for t, p in zip(target.cpu().numpy(), preds.cpu().numpy()):
#                 confusion_stats[int(t)][int(p)] += 1

#         # === original output ===
#         if is_test:
#             classes = range(len(class_emb))
#             cresults = []
#             cresults.append(acc)
#             for c in classes:
#                 cidx = (target == c).nonzero(as_tuple=True)[0]
#                 curr_acc = torch.sum(acc_out[cidx]).item()
#                 cresults.append((c, len(cidx), curr_acc))
#             if return_confusion:
#                 return cresults, preds, confusion_stats
#             return cresults, preds

#         if count_by_batch:
#             result = (torch.sum(acc_out).item(), len(acc_out))
#             if return_confusion:
#                 return result, preds, confusion_stats
#             return result, preds
#         else:
#             if return_confusion:
#                 return acc, preds, confusion_stats
#             return acc, preds
#     else:
#         return None, preds


def contrastive_loss(m_skel, m_txt, m_img, tmp, target, reduction='mean'):
        logits_per_skel_txt = tmp * (m_skel @ m_txt.t())
        logits_per_txt_skel = tmp * (m_txt @ m_skel.t())
        loss = (F.cross_entropy(logits_per_skel_txt, target, reduction=reduction) +
                F.cross_entropy(logits_per_txt_skel, target, reduction=reduction))
        if m_img is not None:
            logits_per_skel_img = tmp * (m_skel @ m_img.t())
            logits_per_img_skel = tmp * (m_img @ m_skel.t())
            loss += (F.cross_entropy(logits_per_skel_img, target, reduction=reduction) +
                    F.cross_entropy(logits_per_img_skel, target, reduction=reduction))
        return loss, logits_per_skel_txt

def accuracy(input, target, class_emb, is_test=False, count_by_batch=False, return_confusion=False):
    expand_input = input.unsqueeze(1).expand([input.shape[0], class_emb.shape[0]] + list(input.shape[1:]))
    expand_class_emb = class_emb.unsqueeze(0).expand([input.shape[0], class_emb.shape[0]] + list(class_emb.shape[1:]))

    final_scores = torch.sum(expand_input * expand_class_emb, dim=-1)
    while len(final_scores.shape) > 2:
        final_scores = final_scores.mean(-1)

    # # 调整温度
    # temperature = 0.1
    # final_scores = final_scores / temperature

    final_scores = F.log_softmax(final_scores, dim=1)
    preds = torch.argmax(final_scores, dim=1)
    
    from collections import Counter
    # print(Counter(preds.cpu().numpy()))

    confusion_stats = defaultdict(Counter)

    if target is not None:
        acc_out = (preds == target)
        acc = torch.sum(acc_out).item() / len(acc_out)

        if return_confusion:
            for t, p in zip(target.cpu().numpy(), preds.cpu().numpy()):
                confusion_stats[int(t)][int(p)] += 1

        if is_test:
            classes = range(len(class_emb))
            cresults = [acc]
            for c in classes:
                cidx = (target == c).nonzero(as_tuple=True)[0]
                curr_acc = torch.sum(acc_out[cidx]).item()
                cresults.append((c, len(cidx), curr_acc))
            return (cresults, preds, confusion_stats) if return_confusion else (cresults, preds)

        result = (torch.sum(acc_out).item(), len(acc_out)) if count_by_batch else acc
        return (result, preds, confusion_stats) if return_confusion else (result, preds)
    else:
        return None, preds
