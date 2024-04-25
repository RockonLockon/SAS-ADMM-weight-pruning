import time
import torch
from net_information import printFun, validate
from torch import optim
import os
from matplotlib import pyplot
from Net_Model import *
from math import ceil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
cat = torch.cat
sign = torch.sign


def to_flat(W):
    W_flat = list(map(lambda x: x.view([-1]), W))
    W_flat = cat(W_flat)
    return W_flat


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def Soft(W, T):
    tmp = (abs(W) - T)
    tmp = (tmp + abs(tmp)) / 2
    y = sign(W) * tmp
    return y

def sasPrun(cnn,paras):
    def to_matrix(W):
        W_matrix = W.split(W_numel)
        W_matrix = list(map(lambda x, y: x.view(y), W_matrix, W_shape))
        return W_matrix

    def get_all_dt(W):
        tmpW = to_matrix(W)
        assign_W(cnn,tmpW)
        out = cnn(img)
        part_cross_entropy = criterion(out, label)
        cnn.zero_grad()
        part_cross_entropy.backward()
        return to_flat(get_W_matrix_grad(cnn)), part_cross_entropy
    beta = paras['beta']
    penalty = paras['lambda_1']
    v = paras['v']
    M = paras['M']
    maxIter = paras['maxIter']
    miniBatch = paras['miniBatch']
    device = paras['device']
    train_dataset = paras['train_dataset']
    filename = paras['file']
    step_par1 = 0.9
    step_par2 = 1.09
    const_sb0 = step_par1 * beta
    const_sbet = step_par2 * beta
    soft_para = penalty / beta
    rho = beta
    M_k = M
    eta_k = 1 / v
    k0 = paras['k0']
    q = paras['q']
    c = M/k0**q
    MkMax = ceil(c*maxIter**q)
    cnn = cnn.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    W_matrix = get_W_matrix(cnn)  # prunFlag 和 zeroFlag正好相反
    W_shape = list(map(lambda x: x.shape, W_matrix))
    W_numel = list(map(lambda x: x.numel(), W_matrix))

    if os.path.exists('checkprune.pth.tar'):
        checkprune = torch.load('checkprune.pth.tar')
        W = checkprune['W']
        Z = checkprune['Z']
        U = checkprune['U']
        W_ag = checkprune['W_ag']
        W_breve = checkprune['W_breve']
        startIter = checkprune['i']
        lossList = checkprune['lossList']
        cnn.load_state_dict(checkprune['state_dict'])
        erg_i = checkprune['erg_i']
    else:
        W = to_flat(W_matrix)
        Z = W.clone()
        U = torch.zeros_like(W)
        W_breve = W.clone()
        lossList = []
        W_W = []
        Z_Z = []
        W_Z = []
        startIter = 1
        erg_i = 1

    diff_WZ = W - Z
    sas_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=miniBatch,
        num_workers=8,
        pin_memory=True,
        sampler = torch.utils.data.RandomSampler(train_dataset, True, MkMax * miniBatch * (maxIter-startIter+1))
    )
    sas_loader = iter(sas_loader)
    start = time.perf_counter()
    # pyplot.ion()
    iteration = 0
    cnn.train()
    criterion.train()
    for i in range(startIter,maxIter+1):
        rho_Wold_h = rho * W + (U - beta * diff_WZ)
        W_old = W
        Z_old = Z
        if i >= k0:
            M_k = ceil(c * i ** q)
            eta_k = M*(M+1) / (v * M_k*(M_k+1))
        const_eta_k = 2 / eta_k
        meanCE = 0
        for t in range(1, M_k + 1):
            beta_t = 2 / (t + 1)
            gamma_t = const_eta_k / t
            tempt_bW = (1 - beta_t) * W

            W_hat = beta_t * W_breve + tempt_bW

            img,label = next(sas_loader)
            img, label = img.to(device), label.to(device)
            d_t,part_cross_entropy = get_all_dt(W_hat)
            temp_gs = gamma_t * W_breve + rho_Wold_h - d_t
            W_breve = temp_gs / (gamma_t + rho)
            W = beta_t * W_breve + tempt_bW
            meanCE += part_cross_entropy
        iteration += M_k
        U_mid = U - const_sb0 * (W - Z)
        soft_Z = W - U_mid / beta
        Z = Soft(soft_Z, soft_para)
        diff_WZ = W - Z
        U = U_mid - const_sbet * diff_WZ
        print('iter{} {:.2f}min: '.format(iteration, (time.perf_counter() - start) / 60), end='')
        print('iter{} {:.2f}min: '.format(iteration, (time.perf_counter() - start)/60), end='',file=open(filename,'a'))
        if i > paras['last']:
            erg_i = erg_i + 1
            alpha_i = 1 / erg_i
            W_ag = (1 - alpha_i) * W_ag + alpha_i * W
        else:
            W_ag = W
        tmpCE = meanCE/M_k
        tmpLoss = (tmpCE + penalty * W_ag.norm(1)).data.item()
        print('Object Value: {:.4f}, Cross Entropy: {:.4f}, W Norm: {:.1f}'.format(tmpLoss, tmpCE.item(),(W_ag.norm(1)).item()), end='')
        print('Object Value: {:.4f}, Cross Entropy: {:.4f}, W Norm: {:.1f}'.format(tmpLoss, tmpCE.item(),(W_ag.norm(1)).item()), end='',file=open(filename,'a'))
        lossList.append(tmpLoss)

        tmpWW = (W - W_old).norm(2).item()**2
        W_W.append(tmpWW)
        tmpZZ = (Z - Z_old).norm(2).item()**2
        Z_Z.append(tmpZZ)
        tmpWZ = (W - Z).norm(2).item()**2
        W_Z.append(tmpWZ)
        print(', WW: {:.4f}, ZZ:{:.4f}, WZ:{:.4f}'.format(tmpWW, tmpZZ, tmpWZ))
        print(', WW: {:.4f}, ZZ:{:.4f}, WZ:{:.4f}'.format(tmpWW,tmpZZ,tmpWZ),file=open(filename,'a'))
        # if i%2==0:
        #     pyplot.clf()
        #     pyplot.plot(lossList)
        #     pyplot.pause(0.2)
        # if i%100==0:
        #     #W U Z i w_ag lossList
        #     saveCheckPrune({'W':W,
        #                     'U':U,
        #                     'Z':Z,
        #                     'W_ag':W_ag,
        #                     'W_breve':W_breve,
        #                     'state_dict': cnn.state_dict(),
        #                     'i':i+1,
        #                     'erg_i':erg_i,
        #                     'lossList':lossList,})

    W_matrix = to_matrix(W_ag)
    assign_W(cnn,W_matrix)
    # pyplot.savefig('tmp.jpg')
    return [W_W,Z_Z,W_Z]

def saveCheckPrune(state,filename='checkprune.pth.tar'):
    torch.save(state,filename)


def sasPrun3(cnn,paras):
    def to_matrix(W):
        W_matrix = W.split(W_numel)
        W_matrix = list(map(lambda x, y: x.view(y), W_matrix, W_shape))
        return W_matrix

    def get_all_dt(W):
        tmpW = to_matrix(W)
        assign_W(cnn,tmpW)
        out = cnn(img)
        part_cross_entropy = criterion(out, label)
        cnn.zero_grad()
        part_cross_entropy.backward()
        return to_flat(get_W_matrix_grad(cnn)), part_cross_entropy


    beta = paras['beta']
    penalty = 0
    v = paras['v']
    M = paras['M']
    maxIter = paras['maxIter']
    miniBatch = paras['miniBatch']
    device = paras['device']
    train_dataset = paras['train_dataset']
    filename = paras['file']
    step_par1 = 0.9
    step_par2 = 1.09
    const_sb0 = step_par1 * beta
    const_sbet = step_par2 * beta
    soft_para = penalty / beta
    rho = beta
    M_k = M
    eta_k = 1 / v
    k0 = paras['k0']
    q = paras['q']
    c = M/k0**q
    MkMax = ceil(c*maxIter**q)
    cnn = cnn.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    W_matrix = get_W_matrix(cnn) 
    W_shape = list(map(lambda x: x.shape, W_matrix))
    W_numel = list(map(lambda x: x.numel(), W_matrix))
    W = to_flat(W_matrix)
    Wzero_index = torch.where(W == 0, 0, 1)

    if os.path.exists('checkretrain.pth.tar'):
        checkRetrain = torch.load('checkretrain.pth.tar')
        W = checkRetrain['W']
        Z = checkRetrain['Z']
        U = checkRetrain['U']
        W_ag = checkRetrain['W_ag']
        W_breve = checkRetrain['W_breve']
        cnn.load_state_dict(checkRetrain['state_dict'])
        startIter = checkRetrain['i']
        erg_i = checkRetrain['erg_i']
        lossList = checkRetrain['lossList']
        testList = checkRetrain['testList']
    else:
        Z = W.clone()
        U = torch.zeros_like(W)
        W_breve = W.clone()
        lossList = []
        testList = []
        startIter = 1
        erg_i = 1

    diff_WZ = W - Z
    sas_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=miniBatch,
        num_workers=8,
        pin_memory=True,
        sampler = torch.utils.data.RandomSampler(train_dataset, True, MkMax * miniBatch * (maxIter-startIter+1))
    )
    sas_loader = iter(sas_loader)
    start = time.perf_counter()
    pyplot.ion()
    iteration = 0
    for i in range(startIter,maxIter+1):
        rho_Wold_h = rho * W + (U - beta * diff_WZ)
        if i >= k0:
            M_k = ceil(c * i ** q)
            eta_k = M*(M+1) / (v * M_k*(M_k+1))
        const_eta_k = 2 / eta_k
        meanCE = 0
        cnn.train()
        criterion.train()
        for t in range(1,M_k+1):
            beta_t = 2 / (t + 1)
            gamma_t = const_eta_k / t
            tempt_bW = (1 - beta_t) * W
            W_hat = beta_t * W_breve + tempt_bW
            img,label = next(sas_loader)
            img, label = img.to(device), label.to(device)
            d_t,part_cross_entropy = get_all_dt(W_hat)
            d_t *= Wzero_index
            temp_gs = gamma_t * W_breve + rho_Wold_h - d_t
            W_breve = temp_gs / (gamma_t + rho)
            W = beta_t * W_breve + tempt_bW
            meanCE += part_cross_entropy
        iteration+=M_k
        U_mid = U - const_sb0 * (W - Z)
        soft_Z = W - U_mid / beta
        Z = Soft(soft_Z, soft_para)
        diff_WZ = W - Z
        U = U_mid - const_sbet * diff_WZ
        print('ITER{} {:.2f}min: '.format(iteration, (time.perf_counter() - start) / 60), end='')
        print('ITER{} {:.2f}min: '.format(iteration, (time.perf_counter() - start)/60), end='',file=open(filename,'a'))
        if i > 150:#600
            erg_i = erg_i + 1
            alpha_i = 1 / erg_i
            W_ag = (1 - alpha_i) * W_ag + alpha_i * W
        else:
            W_ag = W
        tmpCE = meanCE/M_k
        tmpLoss = (tmpCE + penalty*W_ag.norm(1)).data.item()
        lossList.append(tmpLoss)
        print('OV: {:.4f}, CE: {:.4f}, W1: {:.1f}'.format(tmpLoss, tmpCE.item(), (W_ag.norm(1)).item()))
        print('OV: {:.4f}, CE: {:.4f}, W1: {:.1f}'.format(tmpLoss, tmpCE.item(),(W_ag.norm(1)).item()),file=open(filename,'a'))
        if i%10==0:
            tmp = validate(cnn, paras['DCL'])
            testList.append(tmp['top1'].avg)
            print('FakeEpoch{}, top1: {:.6f}%'.format(i, tmp['top1'].avg))
            print('FakeEpoch{}, top1: {:.6f}%'.format(i,tmp['top1'].avg),file = open(filename,'a'))
            pyplot.clf()
            pyplot.subplot(2, 1, 1)
            pyplot.plot(lossList)
            pyplot.subplot(2, 1, 2)
            pyplot.plot(testList)
            pyplot.pause(0.2)
            # saveCheckPrune({'W':W,
            #                 'U':U,
            #                 'Z':Z,
            #                 'W_ag':W_ag,
            #                 'W_breve':W_breve,
            #                 'state_dict': cnn.state_dict(),
            #                 'i':i+1,
            #                 'erg_i':erg_i,
            #                 'lossList':lossList,
            #                 'testList':testList},'checkretrain.pth.tar')
    W_matrix = to_matrix(W_ag)
    assign_W(cnn,W_matrix)
    pyplot.savefig('tmp.jpg')
    print(max(testList))
    print(max(testList),file = open(filename,'a'))

def retrain(cnn,paras):
    def to_matrix(W):
        W_matrix = W.split(W_numel)
        W_matrix = list(map(lambda x, y: x.view(y), W_matrix, W_shape))
        return W_matrix
    device = paras['device']
    modelPath = paras['modelPath']
    dataPath = paras['dataPath']
    filename = paras['file']
    train_dataset, test_dataset, train_loader, test_loader = dataSeting(dataPath, 128, 8, True)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    old_lr = 0.1
    optimizer = optim.SGD(cnn.parameters(), lr=old_lr, momentum=0.9,weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100, 150],last_epoch=-1)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], last_epoch=-1)
    DCL = [device,criterion,test_loader]
    accList = []
    maxC = 0
    W_matrix = get_W_matrix(cnn)
    W_shape = list(map(lambda x: x.shape, W_matrix))
    W_numel = list(map(lambda x: x.numel(), W_matrix))
    W = to_flat(W_matrix)
    Wzero_index = torch.where(W == 0, 0, 1)
    Wzero_index = to_matrix(Wzero_index)
    for epoch in range(1, 200 + 1):
        cnn.train()
        criterion.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = cnn(data)
            loss = criterion(output, target)
            loss.backward()
            W_grad_keep_zero(cnn,Wzero_index)
            optimizer.step()
        value = validate(cnn,DCL)
        accList.append(value['top1'].avg)
        new_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('Train Epoch: {}({:.4f}), loss: {:.5f}, top1: {:.2f}%, ({:.2f}%)'.format(epoch,new_lr,value['losses'].avg, max(accList),value['top1'].avg))
        print('Train Epoch: {}({:.4f}), loss: {:.5f}, top1: {:.2f}%, ({:.2f}%)'.format(epoch,new_lr,value['losses'].avg, max(accList),value['top1'].avg),file=open(filename,'a'))
        lr_scheduler.step()
        if value['top1'].avg > maxC:
            maxC = value['top1'].avg
            torch.save(cnn.state_dict(),modelPath)
            torch.save(optimizer.state_dict(), 'tmpOptimizer')
        # if old_lr != new_lr and value['top1'].avg < maxC:
        #     cnn.load_state_dict(modelPath)
        #     optimizer.load_state_dict(torch.load('tmpOptimizer'))
        #     optimizer.state_dict()['param_groups'][0]['lr'] = new_lr
