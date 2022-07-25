from  __future__ import  absolute_import
from   sklearn.metrics import confusion_matrix
import time
import lib.utils.utils as utils
import torch
import ipdb
import numpy as np
import numpy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None, output_dict=None):

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()

    model.train()

    end = time.time()
    for i, (inp, idx) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        labels = utils.get_batch_label(dataset, idx)
        #print('the labels are',labels);
        #print('the shape are', np.shape(np.array(inp)))
        inp = inp.to(device)

        # inference
        preds = model(inp).cpu()

        # compute loss
        batch_size = inp.size(0)
        #ipdb.set_trace()
        text, length = converter.encode(labels) # length = number of total characteristics in a batch, text = labels
        #ipdb.set_trace()
        preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * batchsize
        loss = criterion(preds, text, preds_size, length)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()

def wer(s1,s2):

    d = numpy.zeros([len(s1)+1,len(s2)+1])
    d[:,0] = numpy.arange(len(s1)+1)
    d[0,:] = numpy.arange(len(s2)+1)

    for j in range(1,len(s2)+1):
        for i in range(1,len(s1)+1):
            if s1[i-1] == s2[j-1]:
                d[i,j] = d[i-1,j-1]
            else:
                d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+1)

    return d[-1,-1]

#def cer(r: list, h: list):
#    """
#    Calculation of CER with Levenshtein distance.
#    """
#    # initialisation
#    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
#    d = d.reshape((len(r) + 1, len(h) + 1))
#    for i in range(len(r) + 1):
#        for j in range(len(h) + 1):
#            if i == 0:
#                d[0][j] = j
#            elif j == 0:
#                d[i][0] = i
#
#    # computation
#    for i in range(1, len(r) + 1):
#        for j in range(1, len(h) + 1):
#            if r[i - 1] == h[j - 1]:
#                d[i][j] = d[i - 1][j - 1]
#            else:
#                substitution = d[i - 1][j - 1] + 1
#                insertion = d[i][j - 1] + 1
#                deletion = d[i - 1][j] + 1
#                d[i][j] = min(substitution, insertion, deletion)
#
#    return d[len(r)][len(h)] / float(len(r)), substitution[len(r)][len(h)],insertion[len(r)][len(h)],deletion[len(r)][len(h)],d[len(r)][len(h)]


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict, output_dict):

    losses = AverageMeter()
    model.eval()

    new_pred = list();
    new_gt   = list();
    n_correct = 0;
    n_correct_new = 0;
    num_test_sample_new = 0;
    with torch.no_grad():
        for i, (inp, idx) in enumerate(val_loader):

            labels = utils.get_batch_label(dataset, idx)
            inp = inp.to(device)
            #ipdb.set_trace()

            # inference
            preds = model(inp).cpu()

            # compute loss
            ##ipdb.set_trace()
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), inp.size(0))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):

                #print( 'the pred length', len(target) )
                #print( 'the target length', pred)
                num_test_sample_new = num_test_sample_new + len(target);
                comp_len = min(len(pred),len(target));
                new_pred.append(str(pred));
                new_gt.append(str(target));

                if pred == target:
                    n_correct += 1
                ##for cn in range(comp_len):
                ##    if pred[cn] == target[cn]:
                ##        n_correct_new +=1;

                n_correct_new  = n_correct_new + len(target) - wer(target,pred);
            if (i + 1) % config.PRINT_FREQ == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))

            if i == config.TEST.NUM_TEST_BATCH:
                break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    #print("the matrix is :", len(pred));
    #print("the matrix is :", len(new_pred));
    #print(confusion_matrix(new_pred,new_gt,normalize='true'))

    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    if num_test_sample > len(dataset):
        num_test_sample = len(dataset)

    print("[#correct:{} / #total:{}]".format(n_correct, num_test_sample))
    #accuracy = n_correct / float(num_test_sample)# modified by jyc
    accuracy = n_correct_new / float(num_test_sample_new)
    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy,new_pred,new_gt
