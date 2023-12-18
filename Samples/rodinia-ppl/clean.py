import os


mainpath = os.getcwd()



for path, directories, files in os.walk('.'):
    directories = [d for d in directories if not d.startswith('.') and not d.startswith('_')]
    if not path.startswith('.\.') and path != '.' and not path.startswith('.\\_'):
        os.chdir(mainpath + '/' + path)
        for file in files:
            if file.endswith(".txt") and file.__contains__("_"):
                os.remove(file)
        os.chdir(mainpath)