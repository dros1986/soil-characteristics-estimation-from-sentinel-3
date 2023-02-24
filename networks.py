import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, conf, ninp=1, nout=1):
        # init
        super(Net, self).__init__()
        # save attributes
        self.conf = conf
        # define number of filters
        nfilters = torch.linspace(self.conf['minf'], self.conf['maxf'], self.conf['nlayers']+1).int()
        nfilters = torch.cat((nfilters, nfilters[:-1].flip(0)),0)
        nfilters = torch.cat((torch.tensor([ninp]),nfilters,torch.tensor([nout])),0)
        # create layers
        blocks = []
        for i in range(1,len(nfilters)):
            blocks.extend( self.block(nfilters[i-1], nfilters[i], bias=True, last=i==len(nfilters)-1) )
        # make it a sequential
        self.blocks = nn.Sequential(*blocks)
        # init weights
        self.init()


    def block(self, inch, outch, bias=True, last=False):
        blocks = []
        # create down block
        blocks.append(torch.nn.Linear(inch.item(), outch.item(), bias=bias))
        # append dropout
        # if not last: blocks.append(nn.Dropout(p=0.2, inplace=True))
        # do not add batchnorm/relu if last layer
        if last: return blocks
        # add batchnorm
        if self.conf['use_batchnorm']:
            if not 'batch_momentum' in self.conf:
                self.conf['batch_momentum'] = 0.1
            blocks.append(nn.BatchNorm1d(outch.item(), momentum=self.conf['batch_momentum']))
        # add non-linearity
        blocks.append(self.get_nonlinearity())
        # add dropout
        if 'dropout' in self.conf and self.conf['dropout'] > 0:
            blocks.append(nn.Dropout(p=self.conf['dropout']))
        # return
        return blocks
    
    
    def get_nonlinearity(self):
        # if leak is defined, use it (for legacy usage)
        if 'leak' in self.conf:
            if self.conf['leak'] > 0:
                return nn.LeakyReLU(negative_slope=self.leak, inplace=True)
            else:
                return nn.ReLU(inplace=True)
            
        # otherwise use parameter non_linearity
        if self.conf['non_linearity'].lower() == 'hardswish':
            return nn.Hardswish(inplace=True)
        if self.conf['non_linearity'].lower() == 'tanhshrink':
            return nn.Tanhshrink()
        if self.conf['non_linearity'].lower() == 'relu':
            return nn.ReLU(inplace=True)
        if self.conf['non_linearity'].lower() == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        
            


    def forward(self, x):
        return self.blocks(x)
    

    def init(self):
        # prepare init function
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                # get the number of the inputs
                # n = m.in_features
                # y = 1.0/math.sqrt(n)
                # m.weight.data.uniform_(-y, y)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        # apply
        self.apply(init_weights)





if __name__ == '__main__':
    # define configuration
    conf = {
        'minf': 3,
        'maxf': 10,
        'nlayers': 5,
        'use_batchnorm':True,
        'batch_momentum':0.1,
        'non_linearity': 'hardswish'
    }
    # create network
    v2v = Net(conf, ninp=1, nout=1)
    v2v.cuda()
    # print it
    print(v2v)
    # print output size
    print(v2v(torch.rand(10,1).cuda()).size())
