import pickle


filename = 'data/wikidata_small/questions/questions.pickle'

questions = pickle.load(open(filename, 'rb'))

num_valid = 5000
num_test = 5000

valid = questions[:num_valid]
test = questions[num_valid: num_valid + num_test]
train = questions[num_test + num_valid:]


data = [test, valid, train]

names = ['test', 'valid', 'train']
prefix = 'data/wikidata_small/questions/'
postfix = '.pickle'

for split, name in zip(data, names):
    filename = prefix + name + postfix
    pickle.dump(split, open(filename, 'wb'))

print('Done')