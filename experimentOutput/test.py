from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir(r'D:\OneDrive\Taif Work\ملفات المواد\عام 2018-2019\Intro. Programming\Lectures') if
             isfile(join(r'D:\OneDrive\Taif Work\ملفات المواد\عام 2018-2019\Intro. Programming\Lectures', f))]

for x in onlyfiles:
    print(x)
