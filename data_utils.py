import pandas as pd
import numpy as np
import tensorflow as tf
import os

def load_and_preprocess_data(csv_path, min_interactions=5):
    r"""
    Loads and preprocesses the Steam reviews dataset.
    Filters users to ensure they have at least min_interactions reviews.
    
    Mathematical Purpose:
    --------------------
    Let U_raw be the set of all users and I_raw be the set of all items in the dataset.
    We define the user interaction sequence for user u as S_u = ( (i_1, t_1), (i_2, t_2), ..., (i_n, t_n) )
    where t_j is the timestamp of interaction j and t_1 < t_2 < ... < t_n.
    
    We filter the set of active users U such that:
        U = { u \in U_raw : |S_u| >= min_interactions }
        
    This ensures that each user has a sequence of length at least `min_interactions`
    to perform leave-one-out splitting:
        - Train: S_u[1 : n-2]
        - Validation Target: S_u[n-1]
        - Test Target: S_n
    """
    print(f"Loading data from {csv_path}...")
    # Load only required columns to save memory
    df = pd.read_csv(csv_path, usecols=['steamid', 'appid', 'timestamp_created', 'review', 'language', 'votes_up', 'weighted_vote_score'])
    
    # Fill missing values
    df['review'] = df['review'].fillna("")
    df['votes_up'] = df['votes_up'].fillna(0)
    df['weighted_vote_score'] = df['weighted_vote_score'].fillna(0.0)
    
    # Sort chronologically globally first to simplify sorting per user
    df['timestamp_created'] = pd.to_datetime(df['timestamp_created'])
    df = df.sort_values(by='timestamp_created').reset_index(drop=True)
    
    # Filter users with at least `min_interactions` interactions
    user_counts = df['steamid'].value_counts()
    active_users = user_counts[user_counts >= min_interactions].index
    
    df_filtered = df[df['steamid'].isin(active_users)].copy()
    
    print(f"Preprocessed dataset: {df_filtered['steamid'].nunique()} active users, {df_filtered['appid'].nunique()} items.")
    return df_filtered

def get_item_descriptions(df):
    r"""
    Extracts a representative English review for each unique item as its description.
    
    Mathematical Purpose:
    --------------------
    For each item i \in I:
        We define the representative semantic text T_i by selecting the review text
        associated with item i that maximizes the weighted vote score in English:
            T_i = argmax_{r \in R_i, lang(r)='english'} ( score(r) )
        where R_i is the set of all reviews for item i.
        If no English review exists, we relax the language constraint.
    """
    print("Extracting representative item descriptions...")
    item_texts = {}
    unique_items = df['appid'].unique()
    
    for appid in unique_items:
        item_df = df[df['appid'] == appid]
        
        # Try finding the top voted English review first
        eng_reviews = item_df[item_df['language'] == 'english']
        if len(eng_reviews) > 0:
            best_review = eng_reviews.loc[eng_reviews['weighted_vote_score'].idxmax()]['review']
        else:
            best_review = item_df.loc[item_df['weighted_vote_score'].idxmax()]['review']
            
        # Clean up review text
        best_review = str(best_review).strip()
        if not best_review or len(best_review) < 3:
            best_review = "Great tactical game with classic multiplayer shooters and community servers."
            
        # Truncate text to avoid excessively long contexts in BERT (limit to 128 words)
        words = best_review.split()
        if len(words) > 128:
            best_review = " ".join(words[:128])
            
        item_texts[appid] = best_review
        
    print(f"Extracted descriptions for all {len(item_texts)} items.")
    return item_texts

def encode_ids(df, item_descriptions):
    r"""
    Maps users (steamid) and items (appid) to sequential integer IDs.
    Reserves sequential ID 0 for padding.
    
    Mathematical Purpose:
    --------------------
    Constructs bijection mappings:
        f_user: U -> {1, 2, ..., |U|}
        f_item: I -> {1, 2, ..., |I|}
    Mapping zero (0) is reserved for padding tokens:
        f_user(pad) = 0, f_item(pad) = 0
    
    This matches embedding layer constraints in TensorFlow where index 0 is used for padding.
    """
    print("Encoding user and item IDs...")
    # Get unique IDs
    unique_users = df['steamid'].unique()
    unique_items = list(item_descriptions.keys())
    
    # Create mappings starting from 1 (0 is reserved for padding)
    user_to_id = {user: idx + 1 for idx, user in enumerate(unique_users)}
    item_to_id = {item: idx + 1 for idx, item in enumerate(unique_items)}
    
    # Map descriptions to sequential item IDs
    id_to_description = {item_to_id[item]: desc for item, desc in item_descriptions.items()}
    # Add padding description at index 0
    id_to_description[0] = "Padding token representation with zero embedding."
    
    df['user_seq_id'] = df['steamid'].map(user_to_id)
    df['item_seq_id'] = df['appid'].map(item_to_id)
    
    return df, user_to_id, item_to_id, id_to_description

def split_and_generate_sequences(df, max_len=10):
    r"""
    Performs Leave-One-Out Time-Based Split and generates input sequences and targets.
    
    Mathematical Purpose:
    --------------------
    For a user interaction sequence S_u = [s_1, s_2, ..., s_n]:
    
    1. Training Set:
       - Input Sequence: S_u^{train} = [s_1, s_2, ..., s_{n-2}]
       - We perform causal autoregressive sequence prediction where:
         Inputs:  X_u^{train} = [s_1, s_2, ..., s_{n-3}]
         Targets: Y_u^{train} = [s_2, s_3, ..., s_{n-2}]
       - This allows the model to compute sequential cross-entropy loss:
         L = - \sum_{t=1}^{T} \log P(Y_u^{train}[t] | X_u^{train}[:t])
       
    2. Validation Set:
       - Input Sequence: X_u^{val} = [s_1, s_2, ..., s_{n-2}]
       - Target: Y_u^{val} = s_{n-1} (the second-to-last item)
       
    3. Test Set:
       - Input Sequence: X_u^{test} = [s_1, s_2, ..., s_{n-1}]
       - Target: Y_u^{test} = s_n (the very last item)
       
    All input sequences are left-padded with 0 to length `max_len`.
    """
    print(f"Splitting data and generating sequences (max_len={max_len})...")
    
    train_seqs, train_targets = [], []
    val_seqs, val_targets = [], []
    test_seqs, test_targets = [], []
    
    # Group by user to extract chronological sequences
    user_groups = df.groupby('user_seq_id')
    
    for user_id, group in user_groups:
        seq = group['item_seq_id'].tolist()
        n = len(seq)
        
        # Since we filtered min_interactions >= 5, n is guaranteed to be >= 5
        
        # 1. Train set: Sequence up to n-2
        # Input sequence: seq[:n-3] -> We shift by 1 to do autoregressive training
        train_input = seq[:n-3]
        train_target = seq[1:n-2]
        
        # Pad train input and target
        padded_train_input = pad_sequence(train_input, max_len)
        padded_train_target = pad_sequence(train_target, max_len)
        
        train_seqs.append(padded_train_input)
        train_targets.append(padded_train_target)
        
        # 2. Validation set: Sequence up to n-2 to predict n-1
        val_input = seq[:n-2]
        val_target = seq[n-2]  # item s_{n-1} (0-indexed: index n-2)
        
        padded_val_input = pad_sequence(val_input, max_len)
        val_seqs.append(padded_val_input)
        val_targets.append(val_target)
        
        # 3. Test set: Sequence up to n-1 to predict n
        test_input = seq[:n-1]
        test_target = seq[n-1]  # item s_n (0-indexed: index n-1)
        
        padded_test_input = pad_sequence(test_input, max_len)
        test_seqs.append(padded_test_input)
        test_targets.append(test_target)
        
    return (np.array(train_seqs, dtype=np.int32), np.array(train_targets, dtype=np.int32),
            np.array(val_seqs, dtype=np.int32), np.array(val_targets, dtype=np.int32),
            np.array(test_seqs, dtype=np.int32), np.array(test_targets, dtype=np.int32))

def pad_sequence(seq, max_len):
    r"""
    Helper function to pad a sequence with zeros to length max_len.
    
    Mathematical Purpose:
    --------------------
    Applies the projection \pi: \mathbb{R}^k -> \mathbb{R}^L:
        If k < L, pre-pad with zeros: [0, ..., 0, s_1, ..., s_k]
        If k >= L, truncate to last L elements: [s_{k-L+1}, ..., s_k]
    """
    if len(seq) < max_len:
        return [0] * (max_len - len(seq)) + seq
    else:
        return seq[-max_len:]

def get_tf_datasets(train_seqs, train_targets, val_seqs, val_targets, test_seqs, test_targets, batch_size=64):
    """
    Creates TensorFlow Datasets for train, val, and test.
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((train_seqs, train_targets))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_seqs, val_targets))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_seqs, test_targets))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset
