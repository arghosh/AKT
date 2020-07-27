import os
import torch
from akt import AKT
# from sakt import SAKT
# from dkvmn import DKVMN
# from dkt import DKT
# from dktplus import DKTPlus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_makedirs(path_):
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass


def get_file_name_identifier(params):
    words = params.model.split('_')
    model_type = words[0]
    if model_type == 'dkt':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_dm', params.d_model], ['_ts', params.train_set],  ['_h', params.hidden_dim], ['_do', params.dropout], ['_l2', params.l2]]
    elif model_type == 'dktplus':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_dm', params.d_model], ['_ts', params.train_set],  ['_h', params.hidden_dim], ['_do', params.dropout], ['_l2', params.l2], ['_r', params.lamda_r], ['_w1', params.lamda_w1], ['_w2', params.lamda_w2]]
    elif model_type == 'dkvmn':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_q', params.q_embed_dim], ['_qa', params.qa_embed_dim], ['_ts', params.train_set], ['_m', params.memory_size], ['_l2', params.l2]]
    elif model_type in {'akt', 'sakt'}:
        file_name = [['_b', params.batch_size], ['_nb', params.n_block], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_do', params.dropout], ['_dm', params.d_model], ['_ts', params.train_set], ['_kq', params.kq_same], ['_l2', params.l2]]
    return file_name


def model_isPid_type(model_name):
    words = model_name.split('_')
    is_pid = True if 'pid' in words else False
    return is_pid, words[0]


def load_model(params):
    words = params.model.split('_')
    model_type = words[0]
    is_cid = words[1] == 'cid'
    if is_cid:
        params.n_pid = -1

    if model_type in {'akt'}:
        model = AKT(n_question=params.n_question, n_pid=params.n_pid, n_blocks=params.n_block, d_model=params.d_model,
                    dropout=params.dropout, kq_same=params.kq_same, model_type=model_type, l2=params.l2).to(device)
    else:
        model = None
    return model
