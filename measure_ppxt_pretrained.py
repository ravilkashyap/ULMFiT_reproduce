import fire
from fastai.text import *
from fastai.lm_rnn import *


def get_language_model(n_tok, em_sz, nhid, nlayers, pad_token, decode_train=True, dropouts=None):
    if dropouts is None: dropouts = [0.5, 0.4, 0.5, 0.05, 0.3]
    rnn_enc = RNN_Encoder(n_tok, em_sz, n_hid=nhid, n_layers=nlayers, pad_token=pad_token,
                          dropouti=dropouts[0], wdrop=dropouts[2], dropoute=dropouts[3], dropouth=dropouts[4])
    rnn_dec = LinearDecoder(n_tok, em_sz, dropouts[1], decode_train=decode_train, tie_encoder=rnn_enc.encoder)
    return SequentialRNN(rnn_enc, rnn_dec)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = 70
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]
    return np.transpose(np.array([data])), target


def measure_ppxt(cuda_id=0):

    torch.cuda.set_device(cuda_id)

    # Trained language model path
    lm_path = 'fwd_wt103'

    # Validation dataset path (idx)
    trn_ids_path = 'data/pretrained/val_ids.npy'
    val_ids_path = 'data/pretrained/val_ids.npy'

    # int -> string vocabulary path (do not use tmp/itos_wt103.pkl; it doesn't match with val_ids.npy)
    itos_path = 'data/pretrained/itos.pkl'

    trn_ids = np.load(trn_ids_path)
    trn_ids = np.concatenate(trn_ids)
    val_ids = np.load(val_ids_path)
    val_ids = np.concatenate(val_ids)

    itos_vocab = pickle.load(open(itos_path, 'rb'))
    stoi_vocab = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos_vocab)})
    vocab_size = len(itos_vocab)

    '''
    for chunk in val_ids:
        for idx, val_id in enumerate(chunk):
            print(itos_vocab[val_id], end=' ')
        print('\n')
    '''

    # Language Model Hyperparameters
    em_sz, nh, nl, bptt, bs = 400, 1150, 3, 70, 1
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*1.0
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

#    data_loader = LanguageModelLoader(val_ids, bs, bptt)


    trn_dl = LanguageModelLoader(trn_ids, bs, bptt)
    val_dl = LanguageModelLoader(val_ids, bs, bptt)
    md = LanguageModelData('data/pretrained/', 1, vocab_size, trn_dl, val_dl, bs=bs, bptt=bptt)

    learner = md.get_model(opt_fn, em_sz, nh, nl,
        dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learner.clip=0.3
    learner.metrics = [accuracy]
    wd=1e-7

    wgts = torch.load('data/pretrained/models/fwd_wt103.h5', map_location=lambda storage, loc: storage)

    print(f'Loading pretrained weights...')
    ew = to_np(wgts['0.encoder.weight'])
    row_m = ew.mean(0)

    itos2 = pickle.load(open('data/pretrained/tmp/itos_wt103.pkl', 'rb'))
    stoi2 = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos2)})
    nw = np.zeros((vocab_size, em_sz), dtype=np.float32)
    nb = np.zeros((vocab_size,), dtype=np.float32)
    for i, w in enumerate(itos_vocab):
        r = stoi2[w]
        if r >= 0:
            nw[i] = ew[r]
        else:
            nw[i] = row_m

    wgts['0.encoder.weight'] = T(nw)
    wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(nw))
    wgts['1.decoder.weight'] = T(np.copy(nw))
    learner.model.load_state_dict(wgts)
    model = learner.model
    criterion = learner.crit

    losses = []

    # Turn on evaluation mode which disables dropout.
    model.reset()
    model.eval()
    total_loss = 0
    #hidden = model.init_hidden(bs)
    hidden = 0
    for i in range(0, len(val_ids)-1, 70):
        data, targets = get_batch(val_ids, i)
        targets = torch.from_numpy(targets).cuda()
        t_data = torch.from_numpy(data).cuda()
        v_data = Variable(t_data)
        prediction = model(v_data)

        if isinstance(prediction,tuple): prediction=prediction[0]

        if prediction.shape[0] != targets.shape[0]:
            zero_append = torch.zeros(1, dtype=torch.int64).cuda()
            targets = torch.cat((targets, zero_append), 0)
        loss = criterion(prediction, targets)

        losses.append(loss.item() * len(data))

        if i % 7000 == 0:
            print('loss at {}/{}: {}\t\tperplexity: {}'.format(i // 7000, len(val_ids) // 7000, loss.item(), 2 ** loss.item()))

    losses = np.asarray(losses)
    final_loss = losses.sum() / len(val_ids)
    print('final loss: {}'.format(final_loss))
    print('final perplexity: {}'.format(2 ** final_loss))

if __name__ == '__main__': fire.Fire(measure_ppxt)
