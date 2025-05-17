import os 
folders =[
    'data', 
    'config',
    'logs',
    'notebook',
    'pipeline',
    'src',
    'static',
    'templates',
    'utils'
   
]

# folders that need __init__.py
init_folders = {
    'config', 'pipeline', 'src', 'utils'
}

#Files to be created at root level
files = [
    'setup.py',
    'requirements.txt',
    'application.py'
]

#create folders
for folder in folders:
    try:
        os.makedirs(folder)
        if folder in init_folders:
            with open(os.path.join(folder, '__init__.py'), 'w') as f:
                f.write("# Auto created __init__.py file")
    except FileExistsError:
        print(f"Folder '{folder}' already exists.")

#create files at root level
for file in files:
    with open(file, 'w') as f:
        f.write("# Auto created file")
        print(f"File '{file}' created.")

print("✅ Project structure created successfully.")