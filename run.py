# Code reused from https://github.com/jennyzhang0215/DKVMN.git
import numpy as np
import torch
import math
from sklearn import metrics
from utils import model_isPid_type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transpose_data_model = {'akt'}


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net, params,  optimizer,  q_data, qa_data, pid_data,  label):
    net.train()
    pid_flag, model_type = model_isPid_type(params.model)
    N = int(math.ceil(len(q_data) / params.batch_size))
    q_data = q_data.T  # Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    # Shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]

    if pid_flag:
        pid_data = pid_data.T
        pid_data = pid_data[:, shuffled_ind]

    pred_list = []
    target_list = []

    element_count = 0
    true_el = 0
    for idx in range(N):
        optimizer.zero_grad()

        q_one_seq = q_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        if pid_flag:
            pid_one_seq = pid_data[:, idx *
                                   params.batch_size:(idx+1) * params.batch_size]

        qa_one_seq = qa_data[:, idx *
                             params.batch_size:(idx+1) * params.batch_size]

        if model_type in transpose_data_model:
            input_q = np.transpose(q_one_seq[:, :])  # Shape (bs, seqlen)
            input_qa = np.transpose(qa_one_seq[:, :])  # Shape (bs, seqlen)
            target = np.transpose(qa_one_seq[:, :])
            if pid_flag:
                # Shape (seqlen, batch_size)
                input_pid = np.transpose(pid_one_seq[:, :])
        else:
            input_q = (q_one_seq[:, :])  # Shape (seqlen, batch_size)
            input_qa = (qa_one_seq[:, :])  # Shape (seqlen, batch_size)
            target = (qa_one_seq[:, :])
            if pid_flag:
                input_pid = (pid_one_seq[:, :])  # Shape (seqlen, batch_size)
        target = (target - 1) / params.n_question
        target_1 = np.floor(target)
        el = np.sum(target_1 >= -.9)
        element_count += el

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        if pid_flag:
            input_pid = torch.from_numpy(input_pid).long().to(device)

        if pid_flag:
            loss, pred, true_ct = net(input_q, input_qa, target, input_pid)
        else:
            loss, pred, true_ct = net(input_q, input_qa, target)
        pred = pred.detach().cpu().numpy()  # (seqlen * batch_size, 1)
        loss.backward()
        true_el += true_ct.cpu().numpy()

        if params.maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=params.maxgradnorm)

        optimizer.step()

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))

        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc


def test(net, params, optimizer, q_data, qa_data, pid_data, label):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    pid_flag, model_type = model_isPid_type(params.model)
    net.eval()
    N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
    q_data = q_data.T  # Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    if pid_flag:
        pid_data = pid_data.T
    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []

    count = 0
    true_el = 0
    element_count = 0
    for idx in range(N):

        q_one_seq = q_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        if pid_flag:
            pid_one_seq = pid_data[:, idx *
                                   params.batch_size:(idx+1) * params.batch_size]
        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx *
                             params.batch_size:(idx+1) * params.batch_size]
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)

        # print 'seq_num', seq_num
        if model_type in transpose_data_model:
            # Shape (seqlen, batch_size)
            input_q = np.transpose(q_one_seq[:, :])
            # Shape (seqlen, batch_size)
            input_qa = np.transpose(qa_one_seq[:, :])
            target = np.transpose(qa_one_seq[:, :])
            if pid_flag:
                input_pid = np.transpose(pid_one_seq[:, :])
        else:
            input_q = (q_one_seq[:, :])  # Shape (seqlen, batch_size)
            input_qa = (qa_one_seq[:, :])  # Shape (seqlen, batch_size)
            target = (qa_one_seq[:, :])
            if pid_flag:
                input_pid = (pid_one_seq[:, :])
        target = (target - 1) / params.n_question
        target_1 = np.floor(target)
        #target = np.random.randint(0,2, size = (target.shape[0],target.shape[1]))

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        if pid_flag:
            input_pid = torch.from_numpy(input_pid).long().to(device)

        with torch.no_grad():
            if pid_flag:
                loss, pred, ct = net(input_q, input_qa, target, input_pid)
            else:
                loss, pred, ct = net(input_q, input_qa, target)
        pred = pred.cpu().numpy()  # (seqlen * batch_size, 1)
        true_el += ct.cpu().numpy()
        #target = target.cpu().numpy()
        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            count += real_batch_size
        else:
            count += params.batch_size

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        # print avg_loss
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    assert count == seq_num, "Seq not matching"

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc
