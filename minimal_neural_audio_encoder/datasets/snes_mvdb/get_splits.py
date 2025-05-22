#######################################################################################
# Split the dataset grouped by how many audios each game has and stratified by genre
#######################################################################################
import os

from priority_group_stratified_split import GroupSet, PrioritySplit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_proportions(group_set:GroupSet):
    proportions = []
    labels = []

    for label, l_groups in group_set.get_label_indexer().items():
        l_proportion = (l_groups.total_size/group_set.total_size)*100
        labels.append(label)
        proportions.append(l_proportion)

    labels_idx = [i for i in range(len(labels))]
    labels_idx.sort(key=lambda idx: labels[idx])
    labels.sort()


    # Sort proportion in labels alphabetic order
    ordered_proportions = [None for _ in range(len(proportions))]

    for i, label_i in (enumerate(labels_idx)):
        ordered_proportions[i] = proportions[label_i]

    return ordered_proportions, labels

def plot_group_set(original:GroupSet, train:GroupSet, eval:GroupSet, test:GroupSet):
    # set width of bar 
    barWidth = 0.2
    fig, ax = plt.subplots(figsize =(12 , 6)) 

    # set height of bar 
    orig_p, orig_l = get_proportions(original)
    eval_p, eval_l = get_proportions(eval)
    test_p, test_l = get_proportions(test)
    train_p, train_l = get_proportions(train)

    orig_l_idx = range(len(orig_l))
    orig_l_idx = sorted(orig_l_idx, key=lambda idx: orig_p[idx], reverse=True)
    orig_l = [orig_l[idx] for idx in orig_l_idx]

    orig_p = sorted(orig_p, reverse=True)
    eval_p = sorted(eval_p, reverse=True)
    test_p = sorted(test_p, reverse=True)
    train_p = sorted(train_p, reverse=True)

    # Set position of bar on X axis 
    br_orig = np.arange(len(orig_p))
    br_eval = [x + barWidth for x in br_orig]
    br_test = [x + barWidth for x in br_eval]
    br_train = [x + barWidth for x in br_test]

    # Add x, y gridlines
    plt.grid(color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.4)

    # Make the plot
    plt.bar(br_orig, orig_p, color ='r', width = barWidth, 
            edgecolor ='grey', label ='Original') 
    plt.bar(br_eval, eval_p, color ='b', width = barWidth, 
            edgecolor ='grey', label ='Eval')
    plt.bar(br_test, test_p, color ='y', width = barWidth, 
            edgecolor ='grey', label ='Test') 
    plt.bar(br_train, train_p, color ='g', width = barWidth, 
            edgecolor ='grey', label ='Train') 


    # Adding Xticks 
    plt.xlabel('Genres', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Segments (%)', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + barWidth for r in range(len(orig_p))], orig_l)
    plt.yticks([x*2 for x in range(7)])
    plt.title("Priority Split: Stratification")

    plt.legend()
    plt.show() 

def save_split_txt(split:GroupSet, name:str):
    games = sorted([game._uid for game in split])
    games = "\n".join(games)

    with open(f'{name}.txt', 'w') as f:
        f.write(games)

    return games

def get_game_soundtrack_df(dataset_path:str, games_genres_csv_path:str) -> pd.DataFrame:
    df_genres = pd.read_csv(games_genres_csv_path)

    df_dict:dict[str, list[str]] = {
        'game': [],
        'genre': [],
        'soundtrack_path': []
    }

    for game_folder in sorted(os.listdir(dataset_path)):
        game_folder_path = os.path.join(dataset_path, game_folder)
        soundtrack_folder_path = os.path.join(game_folder_path, 'soundtracks')

        for soundtrack in sorted(os.listdir(soundtrack_folder_path)):
            soundtrack_path = os.path.join(soundtrack_folder_path, soundtrack)
            genre = df_genres[df_genres['game_folder'] == game_folder]
            genre = genre['game_genre'].tolist()

            if len(genre) == 0:
                print(game_folder, genre)

            df_dict['game'].append(game_folder)
            df_dict['genre'].append(genre[0])
            df_dict['soundtrack_path'].append(soundtrack_path)

    df = pd.DataFrame(df_dict)

    return df

def get_splits(dataset_path:str, games_genres_csv_path:str):
    df = get_game_soundtrack_df(dataset_path, games_genres_csv_path)

    groups = GroupSet.from_df(df, 'game', 'genre')

    priority_split = PrioritySplit()

    splits = priority_split.get_split(groups, [0.8, 0.1, 0.1])

    #train, eval, test = splits
    #plot_group_set(groups, train, eval, test)

    for split, name in zip(splits, ['train', 'eval', 'test']):
        save_split_txt(split, name)

if __name__ == "__main__":
    games_genres = "/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/snes-back/vmdb/deepseek_genres.csv"
    dataset_path = "/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/snes-back/vmdb/nintendo-snes-spc"
    get_splits(dataset_path, games_genres)