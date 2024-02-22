import Pandas as pd

def create_coord_target_files(file, name):
    target_cols=pd.read_csv('data/deepsea_metadata.tsv', sep='\t')['File accession'].tolist() # metadata from build-deepsea-training-dataset repo
    colnames=target_cols+['Chr_No','Start','End']
    df = pd.read_csv(file, usecols=colnames, header=0)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={k:f'y_{k}' for k in target_cols}, inplace=True)
    df.to_csv(f'{name}_coords_targets.csv')


path_to_deepsea_data_repo = sys.argv[1]
create_coord_target_files('debug_valid.tsv', 'val')
create_coord_target_files('debug_test.tsv', 'test')
create_coord_target_files('debug_train.tsv', 'train')