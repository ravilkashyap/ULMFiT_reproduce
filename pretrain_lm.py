import fire
from fastai.text import *

from sampled_sm import *


def train_lm(dir_path, cuda_id, cl=1, bs=64, backwards=False, lr=3e-4, sampled=True,
             pretrain_id=''):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; cl {cl}; bs {bs}; '
          f'backwards {backwards}; lr {lr}; sampled {sampled}; '
          f'pretrain_id {pretrain_id}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)
    PRE = 'bwd_' if backwards else 'fwd_'
    IDS = 'ids'
    p = Path(dir_path)
    assert p.exists(), f'Error: {p} does not exist.'
    bptt=70
    em_sz,nh,nl = 400,1150,3
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    if backwards:
        trn_lm = np.load(p / f'tmp/trn_{IDS}_bwd.npy')
        val_lm = np.load(p / f'tmp/val_{IDS}_bwd.npy')
    else:
        trn_lm = np.load(p / f'tmp/trn_{IDS}.npy')
        val_lm = np.load(p / f'tmp/val_{IDS}.npy')

    ''' trn_lm, val_lm: numericalized datasets
            i.e. trn_lm[0]: [13, 2, 236, ..., 234]  - [I, sent, a, ..., blah]
                 val_lm[0]: [1]                     - [True]
    '''
    trn_lm = np.concatenate(trn_lm)
    val_lm = np.concatenate(val_lm)

    itos = pickle.load(open(p / 'tmp/itos.pkl', 'rb'))
    vs = len(itos)

    trn_dl = LanguageModelLoader(trn_lm, bs, bptt)
    val_dl = LanguageModelLoader(val_lm, bs//5 if sampled else bs, bptt)

    ''' md: model data that stores
            path, pad_idx(?), num tokens, 
            trn data loader, val data loader, 
            batch size, bptt size
    '''
    md = LanguageModelData(p, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

    ''' tprs: normalized word counts as array 
            [323, 23, ..., 12] -> [0.54, 0.12, ..., 0.053]
    '''
    tprs = get_prs(trn_lm, vs)

    ''' drops: dropouts for... 
    '''
    # TODO: fill out what each dropout is for
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.5

    ''' learner: RNN_Learner(Learner) object which wraps model data & nn.Module
                 (Combines ModelData object with nn.Module object - so I can train that module)
    
        crit: CrossEntropyDecoder(nn.Module) (not used as learner has this as attribute)
    '''
    learner,crit = get_learner(drops,    # 5 dropouts
                               15000,    # ?
                               sampled,  # ?
                               md,       # model data
                               em_sz,    # embedding size
                               nh,       # hidden layers size
                               nl,       # num hidden layers
                               opt_fn,   # Adam optimizer
                               tprs)     # nomalized word counts as array
    wd=1e-7
    learner.metrics = [accuracy]

    lrs = np.array([lr/6,lr/3,lr,lr])
    #lrs=lr

    learner.fit(lrs, 1, wds=wd, use_clr=(32,10), cycle_len=cl)
    learner.save(f'{PRE}{pretrain_id}')
    learner.save_encoder(f'{PRE}{pretrain_id}_enc')

if __name__ == '__main__': fire.Fire(train_lm)
