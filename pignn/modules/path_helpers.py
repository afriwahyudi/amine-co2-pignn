# modules/path_helpers.py
import os

def get_path(file_name='earlystop_3.pth', folder_name='models'):
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    model_path = os.path.join(parent_dir, folder_name, file_name)
    return model_path
