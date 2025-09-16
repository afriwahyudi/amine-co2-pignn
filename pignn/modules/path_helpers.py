# modules/path_helpers.py
import os

def get_path(file_name='earlystop_3.pth', folder_name='models'):
    """
    Get the full path to the file by navigating up one directory
    and joining it with a specified folder.
    
    Parameters:
      model_file_name (str): The model file name (default is 'earlystop_3.pth').
      folder_name (str): The folder name where models are stored (default is 'models').
      
    Returns:
      str: The full path to the model file.
    """
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    model_path = os.path.join(parent_dir, folder_name, file_name)
    return model_path
