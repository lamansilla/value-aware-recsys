import numpy as np


def precision_at_k(recommended, relevant, top_L):
    """Computes the precision at Top-L"""
    recommended = recommended[:top_L]
    relevant = set(relevant)

    hits = len(set(recommended) & set(relevant))

    return hits / top_L if top_L > 0 else 0.0


def ndcg_at_k(recommended, relevant, top_L):
    """Computes the NDCG at Top-L"""
    recommended = recommended[:top_L]
    relevant = set(relevant)

    dcg = 0.0
    idcg = 0.0

    for i, item in enumerate(recommended):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2.0)

    for i in range(min(len(relevant), top_L)):
        idcg += 1.0 / np.log2(i + 2.0)

    return dcg / idcg if idcg > 0 else 0.0


def ndcv_at_k(recommended, relevant, scores, top_L):
    """Computes the NDCV at Top-L"""
    recommended = recommended[:top_L]
    relevant = set(relevant)

    dcv = 0.0
    idcv = 0.0

    for i, item in enumerate(recommended):
        if item in relevant:
            dcv += scores[item] / np.log2(i + 2.0)

    ideal_scores = []
    for item in relevant:
        if item in recommended:
            ideal_scores.append(scores[item])
    ideal_scores = sorted(ideal_scores, reverse=True)

    for i in range(min(len(ideal_scores), top_L)):
        idcv += ideal_scores[i] / np.log2(i + 2.0)

    return dcv / idcv if idcv > 0 else 0.0
