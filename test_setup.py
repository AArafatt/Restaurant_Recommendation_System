import os

def test_notebook_folder_exists():
    assert os.path.isdir("notebook"), "Notebook folder does not exist"

def test_streamlit_folder_exists():
    assert os.path.isdir("streamlit"), "Streamlit folder does not exist"

def test_models_folder_exists():
    assert os.path.isdir("models"), "Models folder does not exist"

def test_data_folder_exists():
    assert os.path.isdir("data"), "Data folder does not exist"

def test_requirements_exists():
    assert os.path.isfile("requirements.txt"), "requirements.txt not found"

def test_env_exists():
    assert os.path.isfile(".env"), ".env file not found"

def test_readme_exists():
    assert os.path.isfile("README.md")