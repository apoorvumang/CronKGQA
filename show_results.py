import os

def read5Lines(f):
    lines = []
    for _ in range(5):
        line = f.readline().strip()
        if line != '':
            line = line.split('\t')[:2]
            line = ':'.join(line)
            lines.append(line)
    return lines

def getBestScore(filename, model_name = '',k=1):
    f = open(filename, 'r')
    max_score = 0
    extra_data = []
    s = 'Hits at {k}: '.format(k=k)
    
    for line in f:
        if s in line:
            line = line.strip()
            line = line[len(s):].strip()
            try:
                score = float(line)
            except:
                score = 0.0
            if score > max_score:
                max_score = score
                extra_data = []
                extra_data.append(model_name)
                extra_data.append(line)
                extra_data.extend(read5Lines(f))
                
    return max_score, extra_data

path = 'results/wikidata_big/'
k = 10
score_dict = {}
print('K is %d' %k)
for filename in os.listdir(path):
    if 'ce.log' not in filename:
        continue
    file_path = os.path.join(path, filename)
    model_name = filename.replace('.log', '')
    score, extra_data = getBestScore(file_path, model_name=model_name, k=k)
    # print(filename)
    print(extra_data)
    s = "\tk:{k}\tAccuracy:{score}\tModel: {model}".format(
        k=k,
        model=filename,
        score=score
    )
    score_dict[s] = score
    # print(s)

sd = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1])}

for k, v in sd.items():
    print(k)
# with open(o, 'r') as f: # open in readonly mode
