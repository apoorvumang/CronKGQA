import os


def getBestScore(filename, k=1):
    f = open(filename, 'r')
    scores = []
    s = 'Hits at {k}: '.format(k=k)
    for line in f:
        if s in line:
            line = line.strip()
            line = line[len(s):].strip()
            try:
                score = float(line)
            except:
                score = 0.0
            scores.append(score)
    if len(scores) == 0:
        return 0.0
    return max(scores)

path = 'results/wikidata_big/'
k = 10
score_dict = {}
for filename in os.listdir(path):
    if 'bert' not in filename:
        continue
    file_path = os.path.join(path, filename)
    score = getBestScore(file_path, k=k)
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
