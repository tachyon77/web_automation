import os

from torch import negative, positive


def name_files(source_dir, start):
    id = start
    with os.scandir(source_dir) as it:
        for entry in it:
            if entry.is_file():
                cur_full_path = entry.path
                extension = entry.name.split('.')[-1]
                if extension in ["jpg", "png"]:
                    ordered_name = str(id) + "." + extension
                    ordered_full_path = os.path.join(source_dir, ordered_name)
                    os.rename(cur_full_path, ordered_full_path)
                    id += 1
                
                
 
def gen_train_labels(fileName="trainLabels.csv", positive=0, negative=0):
    f = open(fileName, "w")
    
    for i in range(positive):
        f.write(f"{str(i+1)},yes\n")
    
    for j in range(negative):
        f.write(f"{str(positive+j+1)},no\n")

    f.close()

positive = 124
negative = 41

name_files("/home/mahbub/cs230/project/web_automation/dataset/origin/negative", positive+1)

gen_train_labels(fileName="trainLabels.csv", positive=positive, negative=negative)