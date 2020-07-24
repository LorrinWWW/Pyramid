
import subprocess
import time
import signal, os
from utils import *


process_pool = []

def run_ner(
    model_class = 'PyramidNestNER',
    dataset = 'ACE05',
    pretrained_wv = '../wv/glove.6B.100d.ace05.txt',
    lm_emb_path = '../wv/lm.ace05.pkl',
    token_emb_dim = 100,
    char_emb_dim = 30,
    char_encoder = 'lstm',
    lm_emb_dim = 0,
    bs = 32,
    tag_vocab_size = 60,
    vocab_size = 20000,
    L = 16,
    optim = 'sgd',
    lr = 1e-2,
    dropout = 0.3,
    max_epoches = 500,
    cased = False,
    flag = '0',
):
    
    task_name = f'{model_class}_{dataset}_bs{bs}_drop{dropout}_L{L}_{flag}'
    
    print(f'start running task: {task_name}')
    
    log_file = open(f'./logs/{task_name}.log', 'w')
    model_write_ckpt = f'./ckpts/{task_name}'
    
    arguments = f'''python train_ner.py
        --batch_size {bs}
        --evaluate_interval 500
        --dataset {dataset}
        --pretrained_wv {pretrained_wv} 
        --max_epoches {max_epoches}
        --model_class {model_class} 
        --model_write_ckpt {model_write_ckpt}
        --optimizer {optim}
        --lr {lr}
        --tag_form iob2 
        --cased {int(cased)}
        --token_emb_dim {token_emb_dim}
        --char_emb_dim {char_emb_dim}
        --char_encoder {char_encoder}
        --lm_emb_dim {lm_emb_dim}
        --lm_emb_path {lm_emb_path}
        --tag_vocab_size {tag_vocab_size}
        --vocab_size {vocab_size}
        --dropout {dropout}
        --max_depth {L-1}
    '''
    
    log_file.write(arguments+'\n\n')
    log_file.flush()

    p = subprocess.Popen(arguments.split(), stdin=subprocess.PIPE, stdout=log_file, stderr=log_file)
    
    log_file.write(f'PID: {p.pid}\n\n')
    log_file.flush()
    
    global process_pool
    process_pool.append(p)
    
    time.sleep(60) # let gpu memory usage more stable
    
    return p
    
    
def main():
    
#     for flag in range(5):
        
#         # make sure have enough memory
#         wait_util_enough_mem(3000, sleep_time=30, max_n_try=None)
        
#         run_ner(
#             model_class = 'PyramidNestNER',
#             dataset = 'ACE04',
#             bs = 32,
#             token_emb_dim = 100,
#             pretrained_wv = '../wv/glove.6B.100d.ace04.txt',
#             lm_emb_dim = 0,
#             lm_emb_path = '../wv/lm.ace04.pkl',
#             char_emb_dim = 30,
#             char_encoder = 'lstm',
#             tag_vocab_size = 60,
#             vocab_size = 20000,
#             L = 16,
#             optim = 'sgd',
#             lr = 1e-2,
#             dropout = 0.3,
#             max_epoches = 500,
#             flag = flag,
#         )
    
    for flag in range(5):
        
        # make sure have enough memory
        wait_util_enough_mem(3000, sleep_time=30, max_n_try=None)
        
        run_ner(
            model_class = 'PyramidNestNER',
            dataset = 'ACE05',
            bs = 32,
            token_emb_dim = 100,
            pretrained_wv = '../wv/glove.6B.100d.ace05.txt',
            lm_emb_dim = 0,
            lm_emb_path = '../wv/lm.ace05.pkl',
            char_emb_dim = 30,
            char_encoder = 'lstm',
            tag_vocab_size = 60,
            vocab_size = 20000,
            L = 16,
            optim = 'sgd',
            lr = 1e-2,
            dropout = 0.4,
            max_epoches = 500,
            flag = flag,
        )
        
        
#     for flag in range(5):
        
#         # make sure have enough memory
#         wait_util_enough_mem(5000, sleep_time=30, max_n_try=None)
        
#         run_ner(
#             model_class = 'PyramidNestNER',
#             dataset = 'GENIA',
#             bs = 64,
#             token_emb_dim = 200,
#             pretrained_wv = '../wv/bio_nlp_vec/pubmed_shuffle_win_2_genia.txt',
#             lm_emb_dim = 0,
#             lm_emb_path = '../wv/lm.genia.pkl',
#             char_emb_dim = 60,
#             char_encoder = 'lstm',
#             tag_vocab_size = 60,
#             vocab_size = 30000,
#             L = 16,
#             optim = 'sgd',
#             lr = 1e-2,
#             dropout = 0.4,
#             max_epoches = 200,
#             cased = True, # important
#             flag = flag,
#         )
        
        
        
#     for flag in range(5):
        
#         # make sure have enough memory
#         wait_util_enough_mem(4000, sleep_time=30, max_n_try=None)
        
#         run_ner(
#             model_class = 'PyramidNestNER',
#             dataset = 'NNE',
#             bs = 32,
#             token_emb_dim = 100,
#             pretrained_wv = '../wv/glove.6B.100d.nne.txt',
#             lm_emb_dim = 0,
#             lm_emb_path = '../wv/lm.nne.pkl',
#             char_emb_dim = 30,
#             char_encoder = 'lstm',
#             tag_vocab_size = 400,
#             vocab_size = 50000,
#             L = 10,
#             optim = 'sgd',
#             lr = 1e-2,
#             dropout = 0.2,
#             max_epoches = 500,
#             flag = flag,
#         )
        
        
def handler(signum, frame):
    print('killing all subprocess')
    
    for p in process_pool:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception as e:
            pass
    
    exit()


if __name__ == '__main__':

    signal.signal(signal.SIGINT, handler)
    
    main()
        
    print('Start waiting....')
    for p in process_pool:
        p.wait()
