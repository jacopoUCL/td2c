'''
This file contains the functions to compute the ROC curves and the AUC for the VAR and D2C models.
'''

def precision_top_k(df, top_k):
    '''
    This function computes the precision top k.
    In the context of causal discovery, precision top k is defined as
    the number of true positives in the top k predictions divided by k.

    Since we are dealing with more than just one dags,
    we won't compute the top k in general, since 
    the top k might come all from the same dag.
    
    Therefore, we will take the couple with highest predicted probability 
    of being causal for each graph. And then we will take the top k of those.
    Adapted from https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_References/shared_functions.html#card-precision-top-k-day
    '''

    # Sort the dataframe by predicted_proba after grouping by graph_id and taking the max
    # This is to avoid that all the top k couples come from the same graph
    df = df.sort_values(by="predicted_proba", ascending=False).reset_index(drop=False)
            
    # Get the top k most suspicious couples
    df_top_k=df.head(top_k)
    
    return df_top_k[df_top_k.is_causal==1].shape[0] / top_k