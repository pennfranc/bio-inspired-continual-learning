import numpy as np

def get_score_n_best_consecutive(selected_results, window, score_col):
    selected_results = selected_results.groupby('lr').mean().reset_index().sort_values(by='lr')
    best_idx = 0
    best_score = 0
    for i in range(len(selected_results) - window):
        curr_score = selected_results.iloc[i:i+window]['task_test_accu_last'].mean()
        if curr_score > best_score:
            best_score = curr_score
        best_idx = i

    return best_score, np.std(selected_results.iloc[best_idx:best_idx+window]['task_test_accu_last'])

def get_score_n_after_peak(selected_results, window, score_col, min_accu=None):
    selected_results = selected_results.groupby('lr').mean().reset_index().sort_values(by='lr')
    scores = selected_results[score_col]
    if min_accu:
        max_idx = np.where(np.array(scores) > min_accu)[0][0]
    else:
        max_idx = np.argmax(scores)
    return scores[max_idx:max_idx+window].mean(), scores[max_idx:max_idx+window].std()