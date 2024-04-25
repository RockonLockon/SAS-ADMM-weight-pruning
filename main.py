from torch import optim
from sas import *
from net_information import *
import os
from Net_Model import *

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in [1e-4]:
        for i in [25]:
            paras = {}
            paras['beta'] = 0.04
            paras['lambda_1'] = j
            paras['M'] = 50 # 200
            paras['maxIter'] = 190 # 100
            paras['last'] = 170
            paras['k0'] = 50 # 100
            paras['v'] = i
            paras['miniBatch'] = 256
            paras['q'] = 1.1
            paras['device'] = device
            dataPath = r'E:\2 文档\袁铭的文件\pytorch-sas-pruning\data'
            train_dataset, test_dataset, test_loader = dataSeting2(dataPath, 256, 0, True)
            paras['train_dataset'] = train_dataset

            pa = r'E:\2 文档\袁铭的文件\pytorch-sas-pruning\resNet32-cifar10'
            path = pa + r'\History'
            pathmodel = pa + '\model'
            valueList = [paras['beta'], paras['lambda_1'], paras['maxIter'], paras['v'], paras['k0']]
            nameList = ['beta', 'lambda_1', 'maxIter', 'v', 'k0']
            nameList = [nameList[i] + str(v) for i, v in enumerate(valueList)]
            file = ','.join(nameList)
            paras['file'] = path + '\\' + file + '.txt'
            print(file)
            print(file, file=open(paras['file'], 'a'))
            cnn = resnet32()
            cnn = cnn.to(device)
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.to(device)

            percentList = [85] * 31 + [0]
            DCL = [device, criterion, test_loader]
            # cnn.load_state_dict(torch.load('32maxModel'))
            # printFun(cnn, DCL)
            # sasPrun(cnn, paras)
            # torch.save(cnn.state_dict(), pathmodel + r'\32' + file + '_prune1')
            cnn.load_state_dict(torch.load(pathmodel + r'\32' + file + '_prune1'))
            printFun(cnn, DCL, paras['file'])
            percentPrun(cnn, percentList, device)
            printFun(cnn, DCL, paras['file'])
            paras['DCL'] = DCL
            paras['dataPath'] = dataPath
            paras['modelPath'] = pathmodel + r'\32' + file + '_retrain1'
            retrain(cnn,paras)
            # sasPrun3(cnn, paras)
            # torch.save(cnn.state_dict(), pathmodel + r'\32' + file + '_retrain1')
            # cnn.load_state_dict(torch.load(pathmod,el + r'\32' + file + '_retrain1'))
            # printFun(cnn, DCL)
if __name__ == '__main__':
    main()
