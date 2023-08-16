with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/data--Hive.txt", 'r') as f:
    data_lines = f.readlines()

with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/label--Hive.txt", 'r') as g:
    label_lines = g.readlines()

with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/under--Hive_1-2000.txt", 'r') as p:
    under_lines = p.readlines()

for idx, line in enumerate(under_lines):
    if line.strip() != "None" and line.strip() != "too long.":
        with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/under--Hive_reshaped_.txt", 'a') as t:
            t.write(line)
        with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/data--Hive_reshaped_.txt", 'a') as h:
            h.write(data_lines[idx])
        with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/label--Hive_reshaped_.txt", 'a') as j:
            j.write(label_lines[idx])
