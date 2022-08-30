from utils_tosem import *
import sys
import pickle
from nltk.translate.meteor_score import single_meteor_score
from time import time



class Config:
    def __init__(self):
        self.BATCH_SIZE = 50
        self.EPOCH = 100
        self.MAX_INPUT_SIZE = 400
        self.MAX_OUTPUT_SIZE = 20
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.MODEL_SIZE = 512
        self.DICT_SIZE_1 = 50010
        self.DICT_SIZE_2 = 1010
        self.NUM_LAYER = 1
        self.DROPOUT = 0.25
        self.LR = 1e-4
        
class Config_code2seq:
    def __init__(self):
        self.BATCH_SIZE = 64
        self.EPOCH = 100
        self.MAX_INPUT_SIZE = 400
        self.MAX_OUTPUT_SIZE = 20
        self.START_TOKEN = 0
        self.END_TOKEN = 1
        self.MODEL_SIZE = 256
        self.DICT_SIZE_1 = 30010
        self.DICT_SIZE_2 = 1010
        self.NUM_LAYER = 1
        self.DROPOUT = 0.25
        self.LR = 5e-4
        self.K = 200
        self.SEED = 1234
        self.MAX_SBT_LEN = 20


#func:仅仅提取代码段相关的节点，无依赖
def func(mm):
    ast = mm['ast']
    k = 0
    mask = [True for _ in ast]
    # for i in range(len(ast)):
    #     if ast[k]['num']==ast[i]['num']: k = i
    #     if ast[i]['num']==0: mask[i] = False
    return get_sbt(k,ast,mask)

def func_seq2seq(mm):
    code = nltk.word_tokenize(mm['code'].strip())
    code = [split_word(x) for x in code]
    code = [x for word in code for x in word]
    return code


def func_mix(mm):
    sbt = func(mm)
    code = nltk.word_tokenize(mm['code'].strip())
    code = [split_word(x) for x in code]
    code = [x for word in code for x in word]
    return (sbt,code)

def func_addmethod(mm):
    code = nltk.word_tokenize(mm['code'].strip())
    code = [split_word(x) for x in code]
    code = [x for word in code for x in word]
    ast = mm['ast']
    if ast[0]['num']==0: return (([],[]),[])
    num = [0 for _ in ast]
    for i in range(len(ast)-1,-1,-1):
        num[i] = 1
        for j in ast[i]['children']: num[i] += num[j]
    k = 0
    for i in range(len(ast)):
        if ast[i]['num']==ast[0]['num']: 
            k = i
            if num[i]<=200: break
    sbt = get_sbt(k,ast)
    return (sbt,code)

def func_relate(mm, expand_order=1):
    ast = mm['ast']
    code = nltk.word_tokenize(mm['code'].strip())
    code = [split_word(x) for x in code]
    code = [x for word in code for x in word]
    k = 0
    for i in range(len(ast)):
        if ast[k]['num']==ast[i]['num']: k = i
    mask = [True for _ in ast]
    for i in range(len(ast)):
        if ast[i]['num']==0: mask[i] = False
    li = traverse(k,mm['ast'],mask)
    mask = [False for _ in ast]
    for x in li: mask[x] = True
    for _ in range(expand_order):
        names = find_name(ast,mask)
        scopes = []
        for name in names:
            scope = find_name_scope(ast,name)
            colors = []
            for i in names[name]: colors.append(scope[i])
            for i in range(len(scope)):
                if scope[i] in colors: scope[i] = True
                else: scope[i] = False
            scopes.append((name,scope))
        mask_ = find_scope(ast,scopes)
        mask = [x or y for x,y in zip(mask,mask_)]

    num = [0 for _ in ast]
    for i in range(len(ast)-1,-1,-1):
        node = ast[i]
        num[i] = 1 if mask[i] else 0
        for child in node['children']:
            num[i] += num[child]
    for node in ast:
        i = node['id']
        if num[i]>0:
            mask[i] = True
            if node['type'] == 'ForStatement' or node['type'] == 'WhileStatement' or node['type'] == 'IfStatement':
                child = node['children'][0]
                li = traverse(child,ast)
                for elem in li: mask[elem] = True
    num = [0 for _ in ast]
    for i in range(len(ast)-1,-1,-1):
        node = ast[i]
        num[i] = 1 if mask[i] else 0
        for child in node['children']:
            num[i] += num[child]
    
    root = 0
    for i in range(len(ast)):
        if ast[i]['num']==ast[0]['num']:
            root = i
            if num[i]<=200: break

    relate = get_sbt(root,ast,mask)
    return (relate,code)

def get_batch(path,f,config,in_w2i,out_w2i):
    batch_in1, batch_in2, batch_out = [], [], []
    in1_w2i, in2_w2i = in_w2i
    for tu in get_one(f,path):
       in_, out = tu
       in1, in2 = in_
       in1 = [in1_w2i[x] if x in in1_w2i else in1_w2i['<UNK>'] for x in in1]
       in2 = [in2_w2i[x] if x in in2_w2i else in2_w2i['<UNK>'] for x in in2]
       out = [out_w2i[x] if x in out_w2i else out_w2i['<UNK>'] for x in out]
       in1 = in1[:config.MAX_INPUT_SIZE]
       in2 = in2[:config.MAX_INPUT_SIZE]
       out = out[:config.MAX_OUTPUT_SIZE]
       if len(in1)==0: continue
       if len(in2)==0: continue
       if len(out)==0: continue
       batch_in1.append(in1)
       batch_in2.append(in2)
       batch_out.append(out)
       if len(batch_out)>=config.BATCH_SIZE:
           yield (batch_in1, batch_in2), batch_out
           batch_in1, batch_in2, batch_out = [], [], []
    if len(batch_out)>0:
        yield (batch_in1, batch_in2), batch_out

def get_batch_seq2seq(path,f,config,in_w2i,out_w2i):
    batch_in, batch_out = [], []
    for tu in get_one(f,path):
       in_, out = tu
       in_ = [in_w2i[x] if x in in_w2i else in_w2i['<UNK>'] for x in in_]
       out = [out_w2i[x] if x in out_w2i else out_w2i['<UNK>'] for x in out]
       in_ = in_[:config.MAX_INPUT_SIZE]
       out = out[:config.MAX_OUTPUT_SIZE]
       if len(in_)==0: continue
       if len(out)==0: continue
       batch_in.append(in_)
       batch_out.append(out)
       if len(batch_out)>=config.BATCH_SIZE:
           yield batch_in, batch_out
           batch_in, batch_out = [], []
    if len(batch_out)>0:
        yield batch_in, batch_out

def get_batch_mix(path,f,config,in_w2i,out_w2i):
    batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    in_w2i, in3_w2i = in_w2i
    in1_w2i, in2_w2i = in_w2i
    for tu in get_one(f,path):
       in_, out = tu    #in_是((（type,value),code),nl)
       in_, in3 = in_
       in1,in2 = in_
       in1 = [in1_w2i[x] if x in in1_w2i else in1_w2i['<UNK>'] for x in in1]
       in2 = [in2_w2i[x] if x in in2_w2i else in2_w2i['<UNK>'] for x in in2]
       in3 = [in3_w2i[x] if x in in3_w2i else in3_w2i['<UNK>'] for x in in3]
       out = [out_w2i[x] if x in out_w2i else out_w2i['<UNK>'] for x in out]
       in1 = in1[:config.MAX_INPUT_SIZE]
       in2 = in2[:config.MAX_INPUT_SIZE]
       in3 = in3[:config.MAX_INPUT_SIZE]
       out = out[:config.MAX_OUTPUT_SIZE]
       if len(in1)==0: continue
       if len(in2)==0: continue
       if len(in3)==0: continue
       if len(out)==0: continue
       batch_in1.append(in1)
       batch_in2.append(in2)
       batch_in3.append(in3)
       batch_out.append(out)
       if len(batch_out)>=config.BATCH_SIZE:
           yield ((batch_in1,batch_in2),batch_in3), batch_out
           batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    if len(batch_out)>0:
        yield ((batch_in1,batch_in2),batch_in3), batch_out

def get_batch_addmethod(path,f,config,in_w2i,out_w2i):
    return get_batch_mix(path,f,config,in_w2i,out_w2i)


def get_batch_relate(path,f,config,in_w2i,out_w2i):
    return get_batch_mix(path,f,config,in_w2i,out_w2i)

def get_batch_relate_api(code_path,api_path,f,config,in_w2i,out_w2i):     #in_w2i=((type_w2i,value_w2i),code_w2i, api_w2i),code_path里有
    batch_in1, batch_in2, batch_in3, batch_in4, batch_out = [], [], [], [], []     #type,value,code,api,comment
    in_w2i, in3_w2i, in4_w2i = in_w2i
    in1_w2i, in2_w2i = in_w2i
    with open (api_path,'r') as f1:
        for tu, api_token in zip(get_one(f, code_path),f1):
            words = nltk.word_tokenize(api_token)
            words = [split_word(word) for word in words]
            in4 = [x for word in words for x in word]
            in_, out = tu  # in_是((（type,value),code),nl)
            in_, in3 = in_
            in1, in2 = in_
            in1 = [in1_w2i[x] if x in in1_w2i else in1_w2i['<UNK>'] for x in in1]
            in2 = [in2_w2i[x] if x in in2_w2i else in2_w2i['<UNK>'] for x in in2]
            in3 = [in3_w2i[x] if x in in3_w2i else in3_w2i['<UNK>'] for x in in3]
            in4 = [in4_w2i[x] if x in in4_w2i else in4_w2i['<UNK>'] for x in in4]
            out = [out_w2i[x] if x in out_w2i else out_w2i['<UNK>'] for x in out]
            in1 = in1[:config.MAX_INPUT_SIZE]
            in2 = in2[:config.MAX_INPUT_SIZE]
            in3 = in3[:config.MAX_INPUT_SIZE]
            in4 = in4[:config.MAX_INPUT_SIZE]
            out = out[:config.MAX_OUTPUT_SIZE]
            if len(in1) == 0: continue
            if len(in2) == 0: continue
            if len(in3) == 0: continue
            if len(in4) == 0: continue
            if len(out) == 0: continue
            batch_in1.append(in1)
            batch_in2.append(in2)
            batch_in3.append(in3)
            batch_in4.append(in4)
            batch_out.append(out)
            if len(batch_out) >= config.BATCH_SIZE:
                yield ((batch_in1, batch_in2), batch_in3, batch_in4), batch_out
                batch_in1, batch_in2, batch_in3, batch_in4, batch_out = [], [], [], [], []
        if len(batch_out) > 0:
            yield ((batch_in1, batch_in2), batch_in3, batch_in4), batch_out

if __name__ == "__main__":
    # model_name = sys.argv[1]
    model_name = 'code2seq'
    datatype = 'method'
    start = -1
    path = ''   # path for dataset
    config = Config()
    logger = get_logger('{}/{}_logging_{}_dict3w.txt'.format(path,model_name,datatype))
    code_w2i = pickle.load(open('{}/dict3w/{}.dict.code.w2i'.format(path,datatype),'rb'))
    ast_w2i = pickle.load(open('{}/dict3w/{}.dict.ast.w2i'.format(path,datatype),'rb'))
    type_w2i = pickle.load(open('{}/dict3w/{}.dict.type.w2i'.format(path,datatype),'rb'))
    value_w2i = pickle.load(open('{}/dict3w/{}.dict.value.w2i'.format(path,datatype),'rb'))
    nl_w2i = pickle.load(open('{}/dict3w/{}.dict.nl.w2i'.format(path,datatype),'rb'))
    code_i2w = pickle.load(open('{}/dict3w/{}.dict.code.i2w'.format(path,datatype),'rb'))
    ast_i2w = pickle.load(open('{}/dict3w/{}.dict.ast.i2w'.format(path,datatype),'rb'))
    type_i2w = pickle.load(open('{}/dict3w/{}.dict.type.i2w'.format(path,datatype),'rb'))
    value_i2w = pickle.load(open('{}/dict3w/{}.dict.value.i2w'.format(path,datatype),'rb'))
    nl_i2w = pickle.load(open('{}/dict3w/{}.dict.nl.i2w'.format(path,datatype),'rb'))

    if model_name == 'deepcom':
        from deepcom import *
        in_w2i = (type_w2i,value_w2i)
        get_batch = get_batch
        f = func
        model = Model(config)
        if start != -1:
            model.load('{}/model/{}_{}'.format(path, model_name, start), model_name)

    elif model_name == 'seq2seq':
        from seq2seq import *
        in_w2i = code_w2i
        get_batch = get_batch_seq2seq
        f = func_seq2seq
        model = Model(config)
    elif model_name == "code2seq":
        from code2seq.code2seq import *
        # from code2seq.config import Config
        # from utils_tosem import RandomPathDataGen
        config = Config_code2seq()
        model = Code2Seq(config)
        train_gen = RandomPathDataGen(path,config.K)
        valid_gen = RandomPathDataGen(path,config.K)
        test_gen = RandomPathDataGen(path,config.K)
        if start != -1:
            model.load("{}/model/{}_{}.pkl".format(path, model_name, start))  # 如果模型有的话就读取模型
            model.set_trainer()
    out_w2i = nl_w2i
    best_bleu = 0.
    if model_name != 'code2seq':
        for epoch in range(start + 1, config.EPOCH):
            loss = 0.
            for step,batch in enumerate(get_batch('{}/{}.train'.format(path,datatype),f,config,in_w2i,out_w2i)):
                batch_in, batch_out = batch
                loss += model(batch_in,True,batch_out)
                logger.info('Epoch: {}, Batch: {}, Loss: {}'.format(epoch,step,loss/(step+1)))
                # if step == 10:
                #     break
            preds = []
            refs = []
            bleu1 = bleu2 = bleu3 = bleu4 = meteor = rouge = 0.
            len_rouge_preds = 0
            for step,batch in enumerate(get_batch('{}/{}.valid'.format(path,datatype),f,config,in_w2i,out_w2i)):
                batch_in, batch_out = batch
                pred = model(batch_in,False)
                preds += pred
                refs += batch_out
                len_rouge_preds += len(pred)
                for x,y in zip(batch_out,pred):
                    bleu1 += calc_bleu([x],[y],1)
                    bleu2 += calc_bleu([x], [y], 2)
                    bleu3 += calc_bleu([x], [y], 3)
                    bleu4 += calc_bleu([x], [y], 4)
                    meteor += single_meteor_score(' '.join([str(z) for z in x]),' '.join([str(z) for z in y]))
                    if len(y) > 0:
                        rouge += myrouge([x],y)
                    else:
                        len_rouge_preds -= 1
                logger.info('Epoch: {}, Batch: {}, BLEU-1: {}, BLEU-2: {},BLEU-3: {}, BLEU-4: {}, METEOR: {}, ROUGE-L: {}'.format(epoch,step,bleu1/len(preds),bleu2/len(preds),bleu3/len(preds),bleu4/len(preds),meteor/len(preds),rouge/len_rouge_preds))
            logger.info('Epoch: {}, Batch: {}, BLEU-1: {}, BLEU-2: {},BLEU-3: {}, BLEU-4: {}, METEOR: {}, ROUGE-L: {}'.format(epoch, step, bleu1/len(preds), bleu2/len(preds), bleu3/len(preds), bleu4/len(preds),meteor/len(preds), rouge/len_rouge_preds))

            if bleu4>best_bleu:
                best_bleu = bleu4
                model.save('{}/model/{}_{}'.format(path,model_name,epoch),model_name)
            preds = []
            refs = []
            bleu1 = bleu2 = bleu3 = bleu4 = meteor = rouge = 0.
            len_rouge_preds = 0
            for step,batch in enumerate(get_batch('{}/{}.test'.format(path,datatype),f,config,in_w2i,out_w2i)):
                batch_in, batch_out = batch
                pred = model(batch_in,False)
                preds += pred
                refs += batch_out
                len_rouge_preds += len(pred)
                for x,y in zip(batch_out,pred):
                    bleu1 += calc_bleu([x], [y], 1)
                    bleu2 += calc_bleu([x], [y], 2)
                    bleu3 += calc_bleu([x], [y], 3)
                    bleu4 += calc_bleu([x], [y], 4)
                    meteor += single_meteor_score(' '.join([str(z) for z in x]),' '.join([str(z) for z in y]))
                    if len(y) > 0:
                        rouge += myrouge([x],y)
                    else:
                        len_rouge_preds -= 1
                logger.info('Epoch: {}, testBatch: {}, BLEU-1: {}, BLEU-2: {},BLEU-3: {}, BLEU-4: {}, METEOR: {}, ROUGE-L: {}'.format(epoch, step, bleu1 / len(preds), bleu2 / len(preds), bleu3 / len(preds), bleu4 / len(preds), meteor / len(preds), rouge / len_rouge_preds))
            logger.info('Epoch: {}, testBatch: {}, BLEU-1: {}, BLEU-2: {},BLEU-3: {}, BLEU-4: {}, METEOR: {}, ROUGE-L: {}'.format(epoch, step, bleu1 / len(preds), bleu2 / len(preds), bleu3 / len(preds), bleu4 / len(preds), meteor / len(preds), rouge / len_rouge_preds))


    elif model_name == 'code2seq':
        t0 = time()
        for epoch in range(start+1,config.EPOCH):
        #'''
            loss = []
            has_trained = 0
            batch_num = 0
            for batch in train_gen.get_iter(config.BATCH_SIZE, code_w2i, ast_w2i, nl_w2i, config.MAX_INPUT_SIZE,config.MAX_SBT_LEN, config.MAX_OUTPUT_SIZE,'train'):
                batch_num += 1
                batch_code, batch_sbt, batch_nl = batch #在code2seq中batch_sbt就是ast的路径
                result = model(batch_sbt, True, batch_nl)
                loss.append(result)
                logger.info("Epoch={}/{}, Batch={}, train_loss={}, time={}".format(epoch,config.EPOCH,batch_num,loss[-1],get_time(time()-t0)))
                # if batch_num == 5:break
            #'''


            preds = []
            refs = []
            has_valid = 0
            for batch in valid_gen.get_iter(config.BATCH_SIZE, code_w2i, ast_w2i, nl_w2i, config.MAX_INPUT_SIZE,config.MAX_SBT_LEN, config.MAX_OUTPUT_SIZE,'valid'):
                has_valid += 1
                batch_code, batch_sbt, batch_nl = batch
                pred = model(batch_sbt, False)
                preds += pred
                refs += batch_nl
                print(has_valid)
                # if has_valid == 5: break
            score = calc_bleu(refs, preds)  # 他们计算bleu的方法，用nltk的api
            logger.info("Epoch={}/{}, valid_sentence_bleu={}, time={}".format(epoch, config.EPOCH,score,get_time(time() - t0)))
            if score>best_bleu:
                best_bleu = score
                logger.info("Saving the model {} with the best bleu score {}\n".format(epoch, best_bleu))   #取最优的保存
                model.save('{}/{}_{}.pkl'.format(path,model_name,epoch))