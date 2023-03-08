import pandas as pd 

def calculate_F2score(pred_df: pd.DataFrame, act_df: pd.DataFrame):
    
    """
    Using predictions_df and actual_df as exploded correlation columns to calculate F1 score.
    Results show correct predicts, recall, precision and F2 score.
    Results also return the list of correct predicts, correct_df_
    """
    print ('\nCalculating scores...')
    correct_preds=[]
    correct_pairs=[]
    if pred_df.empty or act_df.empty:
        print ('\nOne or both dataframes are empty. Abort F2score calculation.')
        return None
    prediction_df=pred_df.copy()
    actual_df = act_df.copy()
    prediction_df.columns=['topic_id', 'content_ids_pred']
    actual_df.columns=['topic_id', 'content_ids_actual']
    df = pd.merge(prediction_df, actual_df, how='inner', on='topic_id')
    if df.empty:
        print ('\nNo matches between predictions and correlations. Abort F2score calculation.')
        return None
    for row in df.itertuples():
        counts = 0
        for id in row.content_ids_pred.split(' '):
            correct_pairs.append([row.topic_id, id])
            if id in row.content_ids_actual.split(' '):
                counts += 1 
        correct_preds.append (counts)
    df['correct_pred'] = correct_preds
    df['precision'] = df['correct_pred']/(df.content_ids_actual.str.len() + 1e-7)
    df['recall'] = df['correct_pred']/(df.content_ids_pred.str.len() + 1e-7)
    for beta in [0.5, 1, 2]:
        df['f'+str(beta)] = ((1 + beta**2) * df['precision'] * df['recall'])/((beta**2 * df['precision']) + df['recall'] + 1e-7) 
    print ('\nF2 score calculation finished.')

    return df, correct_pairs