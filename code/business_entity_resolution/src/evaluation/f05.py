def entity_f05(true_ids, pred_ids):
    true_ids = set(true_ids) if true_ids else set()
    pred_ids = set(pred_ids) if pred_ids else set()
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids and pred_ids:
        return 0.0
    if not pred_ids and true_ids:
        return 0.0
    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids) if pred_ids else 0.0
    recall = tp / len(true_ids) if true_ids else 0.0
    if precision==0 or recall==0:
        return 0.0
    return (1.25*precision*recall)/(0.25*precision+recall)

def macro_f05(gt_dict, pred_dict, s1_ids):
    scores=[]
    for sid in s1_ids:
        scores.append(entity_f05(gt_dict.get(sid, set()), pred_dict.get(sid, set())))
    return sum(scores)/len(scores) if scores else 0.0

def parse_gt(gt_df):
    d={}
    for _,row in gt_df.iterrows():
        sid=row["source1_entity_id"]
        mids=row["matched_entity_ids"]
        if not isinstance(mids,str) or mids.strip()=="":
            d[sid]=set()
        else:
            d[sid]=set([x.strip() for x in mids.split(",") if x.strip()])
    return d
