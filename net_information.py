import torch
from Net_Model import *
import numpy as np
import sas1 as sas


def zeroNum(W):
    return len(torch.where(W == 0)[0])


def eleNum(W):
    return torch.sum(torch.where(W == 0, 1, 1)).item()


def printFun(cnn, DCL ,file = None):
    W_matrix = get_W_matrix(cnn)
    W = sas.to_flat(W_matrix)
    print('W norm1:', W.norm(1).item())
    print('Num of Nonzero:', eleNum(W) - zeroNum(W))
    print('Num of Total:', eleNum(W))
    print('Pro of Nonzero: {:.3f}%'.format((1 - zeroNum(W) / eleNum(W)) * 100))
    print('      ', end='')
    value = validate(cnn,DCL)
    print('top1: {:.2f}'.format(value['top1'].avg))
    if file != None:
        print('W norm1:', W.norm(1).item(), file = open(file, 'a'))
        print('Num of Nonzero:', eleNum(W) - zeroNum(W), file = open(file, 'a'))
        print('Num of Total:', eleNum(W), file = open(file, 'a'))
        print('Pro of Nonzero: {:.3f}%'.format((1 - zeroNum(W) / eleNum(W)) * 100), file = open(file, 'a'))
        print('      ', end='', file = open(file, 'a'))
        print('top1: {:.2f}'.format(value['top1'].avg), file = open(file, 'a'))
    return value

def printModelFun(filename, cnn, percentlist):
    cnn.load_state_dict(torch.load(filename))
    print(filename)
    W_matrix = get_W_matrix(cnn)
    allNum = eleNum(sas.to_flat(W_matrix))
    for i in range(len(W_matrix)):
        tmpNum = len(W_matrix[i].view(-1))
        print('  Layer {}-th contain {} weights ({:.3f}%)'.format(i + 1, tmpNum, tmpNum / allNum * 100))
    print('  Before Training:')
    W = sas.to_flat(get_W_matrix(cnn))
    printFun(W)

    print('  pruning后:')
    percentPrun(get_W_matrix(cnn), percentlist)
    W = sas.to_flat(get_W_matrix(cnn))
    printFun(W)

    print('  训练后:')
    trainForPruning(cnn, get_W_matrix(cnn), traintimes=30, printspan=600, corretFlag=True)  # 15times
    W = sas.to_flat(get_W_matrix(cnn))
    printFun(W)

def testFun(cnn, DOCDD, printFlag=True):
    [device, optimizer, criterion, train_loader, test_loader] = DOCDD
    cnn.eval()
    criterion.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = cnn(data)
            loss = criterion(output, target)
            print_loss = loss.data.item()
            test_loss += print_loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if printFlag == True:
        print('average loss: {:.4f}, correct: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update(self, sum, n):
        self.sum += sum
        self.count += n
        self.avg = 0 if self.count == 0 else self.sum / self.count * 100.0

def generate_map():
    return {i: 0 for i in range(10)}

def Accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
    num_map = generate_map()
    correct_map = generate_map()
    n_correct = correct[:k].view(-1)
    for i in range(batch_size):
        t = target[i].item()
        num_map[t] += 1
        if n_correct[i].item():
            correct_map[t] += 1
    return res, num_map, correct_map

def validate(cnn, DCL, printFlag=True):
    [device, criterion, test_loader] = DCL
    """
    Run evaluation
    """
    cnn.eval()
    criterion.eval()
    t = {'losses': AverageMeter(),
         'top1': AverageMeter()}
    for i in range(10):
        t[i] = AverageMeter()
    total_steps = len(test_loader)
    iterats = iter(test_loader)
    for step in range(total_steps):
        input, target = next(iterats)
        target = target.to(device)
        input_var = input.to(device)
        target_var = target
        output = cnn(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1, num_map, correct_map = Accuracy(output.data, target)
        t['losses'].update(loss.item(), input.size(0))
        t['top1'].update(sum=prec1[0].item(), n=input.size(0))
        for j in range(10):
            t[j].update(sum=correct_map[j], n=num_map[j])
    return t

def percentPrun(cnn, percentList, device):
    W_matrix = get_W_matrix(cnn)
    new_W_matrix = []
    count = 0
    for w in W_matrix:
        w = w.cpu()
        wnp = w.numpy()
        pcen = np.percentile(abs(wnp), percentList[count])
        under_threshold = abs(wnp) < pcen
        wnp[under_threshold] = 0
        # above_threshold = abs(wnp) >= pcen
        new_W_matrix.append(torch.tensor(wnp).to(device))
        count += 1
    assign_W(cnn,new_W_matrix)
