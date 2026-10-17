"""
Official Evaluation Metric: Macro-averaged F_0.5 across all Source 1 entities.
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
"""

def compute_macro_f05(ground_truth_dict, predictions_dict, all_s1_ids=None):
    if all_s1_ids is None:
        all_s1_ids = list(ground_truth_dict.keys())
        
    total_entities = len(all_s1_ids)
    if total_entities == 0:
        return {"macro_f05": 0.0, "total": 0}
        
    total_f05 = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    singleton_total = 0
    singleton_correct = 0
    non_singleton_total = 0
    non_singleton_f05_sum = 0.0
    pred_non_empty_count = 0
    
    for s1_id in all_s1_ids:
        gt_set = ground_truth_dict.get(s1_id, set())
        pred_set = predictions_dict.get(s1_id, set())
        
        if len(pred_set) > 0:
            pred_non_empty_count += 1
            
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if len(gt_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                entity_f05 = 1.0
            else:
                entity_f05 = 0.0
        else:
            non_singleton_total += 1
            if len(pred_set) == 0:
                entity_f05 = 0.0
            else:
                prec = tp / len(pred_set)
                rec = tp / len(gt_set)
                denom = 0.25 * prec + rec
                if denom > 0:
                    entity_f05 = (1.25 * prec * rec) / denom
                else:
                    entity_f05 = 0.0
            non_singleton_f05_sum += entity_f05
            
        total_f05 += entity_f05
        
    macro_f05 = total_f05 / total_entities
    micro_prec = total_tp / max(1, total_tp + total_fp)
    micro_rec = total_tp / max(1, total_tp + total_fn)
    
    return {
        "macro_f05": macro_f05,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "total_s1": total_entities,
        "singletons": singleton_total,
        "singletons_correct": singleton_correct,
        "singleton_accuracy": singleton_correct / max(1, singleton_total),
        "non_singleton_total": non_singleton_total,
        "non_singleton_f05": non_singleton_f05_sum / max(1, non_singleton_total),
        "predicted_with_matches": pred_non_empty_count,
        "predicted_singletons": total_entities - pred_non_empty_count,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn
    }
