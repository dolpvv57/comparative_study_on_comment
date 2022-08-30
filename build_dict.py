import json
from utils_tosem import *
from collections import defaultdict,Counter
import nltk
import pickle

def build_dict(mode):
    nl_dict = defaultdict(int)
    type_dict = defaultdict(int)
    value_dict = defaultdict(int)
    code_dict = defaultdict(int)
    num = 0
    path = ""    # path for dataset
    with open('{}/{}.train'.format(path, mode)) as f:
        for line in f:
            num += 1
            mm = json.loads(line)
            code = mm['code']
            nl = mm['nl']
            ast = mm['ast']

            code = nltk.word_tokenize(code.strip())
            code = [split_word(x) for x in code]
            code = [x for word in code for x in word]
            for subword in code:
                code_dict[subword] += 1

            for node in ast:
                _type = node['type']
                _val = node['value']
                type_dict[_type] += 1
                for subword in nltk.word_tokenize(' '.join(split_word(_val))):
                    value_dict[subword] += 1

            for word in nltk.word_tokenize(nl):
                words = ' '.join(split_word(word))
                words = normalize_word(words)
                for subword in nltk.word_tokenize(' '.join(words)):
                    nl_dict[subword] += 1
            if num % 1000 == 0: print(num)
    
    print(len(code_dict), len(type_dict),len(value_dict),len(nl_dict))
    code_dict = ['<S>','</S>','<UNK>'] + [x[0] for x in Counter(code_dict).most_common(50000)]
    type_dict = ['<UNK>'] + [x[0] for x in Counter(type_dict).most_common(50000)]
    value_dict = ['<UNK>','<SUB>','</SUB>'] + [x[0] for x in Counter(value_dict).most_common(50000)]
    nl_dict = ['<S>','</S>','<UNK>'] + [x[0] for x in Counter(nl_dict).most_common(50000)]

    with open('./data/{}.dict.code.w2i'.format(mode),'wb') as f:
        mm = {w:i for i,w in enumerate(code_dict)}
        pickle.dump(mm,f)

    with open('./data/{}.dict.code.i2w'.format(mode),'wb') as f:
        mm = {i:w for i,w in enumerate(code_dict)}
        pickle.dump(mm,f)

    with open('./data/{}.dict.type.w2i'.format(mode),'wb') as f:
        mm = {w:i for i,w in enumerate(type_dict)}
        pickle.dump(mm,f)

    with open('./data/{}.dict.type.i2w'.format(mode),'wb') as f:
        mm = {i:w for i,w in enumerate(type_dict)}
        pickle.dump(mm,f)

    with open('./data/{}.dict.value.w2i'.format(mode),'wb') as f:
        mm = {w:i for i,w in enumerate(value_dict)}
        pickle.dump(mm,f)

    with open('./data/|{}.dict.value.i2w'.format(mode),'wb') as f:
        mm = {i:w for i,w in enumerate(value_dict)}
        pickle.dump(mm,f)

    with open('./data/{}.dict.nl.w2i'.format(mode),'wb') as f:
        mm = {w:i for i,w in enumerate(nl_dict)}
        pickle.dump(mm,f)

    with open('./data/{}.dict.nl.i2w'.format(mode),'wb') as f:
        mm = {i:w for i,w in enumerate(nl_dict)}
        pickle.dump(mm,f)

def bulid_ast_dict(mode):
    ast_dict = defaultdict(int)
    path = ""    # path for dataset
    with open('{}/{}_train.sbt'.format(path, mode)) as f:
        num = 0
        for line in f.readlines():
            num += 1
            line = line.split()
            for token in line:
                ast_dict[token] += 1
    print(len(ast_dict))
    ast_dict = ['<UNK>','<SUB>','</SUB>'] + [x[0] for x in Counter(ast_dict).most_common(50000)]
    with open('{}/{}.dict.ast.w2i'.format(path, mode),'wb') as f:
        mm = {w:i for i,w in enumerate(ast_dict)}
        pickle.dump(mm,f)

    with open('{}/{}.dict.ast.i2w'.format(path, mode),'wb') as f:
        mm = {i:w for i,w in enumerate(ast_dict)}
        pickle.dump(mm,f)


if __name__ == "__main__":

    build_dict('method')

