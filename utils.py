import torch
import torch.nn as nn
import collections
import random
import sys
import time
import numpy as np
import wandb
import scipy.stats as st
device = "cuda" if torch.cuda.is_available() else "cpu"
bs = 100
total_n_epoch_ = 120
number_of_runs = 10
### compute config

def compute_config(dict_config):
    try:
        dataset = dict_config['dataset']
    except:
        dataset = "IBC"
    print("Working with dataset", dataset)
    print("using", device)
    
    print("loading dataset...", end = '')
    x_train = torch.load("X_train_" + dataset + ".pt").to(device).float()
    x_val = torch.load("X_val_" + dataset + ".pt").to(device).float()
    x_test = torch.load("X_test_" + dataset + ".pt").to(device).float()
    y_train = torch.load("Y_train_" + dataset + ".pt").to(device).long()
    y_val = torch.load("Y_val_" + dataset + ".pt").to(device).long()
    y_test = torch.load("Y_test_" + dataset + ".pt").to(device).long()
    print("sizes train", x_train.shape[0], "val", x_val.shape[0], "test", x_test.shape[0])
    
    eigs = torch.zeros(1,1).to(device)
    if dict_config['model']=='mlp' and dict_config['filtering'] is None:
        graph = None
        print("using no graphs")
    else :
        if dict_config['graph']=='anatomical':
            graph = torch.load("anatomic_graph.pt").to(device)
            print("using anatomical graph")
        elif dict_config['graph']=='functional':
            graph = torch.load("functional_graph.pt").to(device)
            print("using functional graph")
    
    mean = x_train.mean(dim = 0)
    std = x_train.std(dim = 0)
    x_train, x_val, x_test = (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std
    
    #if dict_config['model'] == 'mlp' and dict_config['filtering'] is not None:
    #    graph = torch.matmul(x_train.squeeze(1).transpose(0,1), x_train.squeeze(1))
    
    if graph is not None:
        knn_graph = torch.zeros(graph.shape).to(device)
        for i in range(knn_graph.shape[0]):
            graph[i,i] = 0
            best_k = torch.sort(graph[i,:])[1][-8:]
            knn_graph[i, best_k] = 1
            knn_graph[best_k, i] = 1
        degree = torch.diag(torch.pow(knn_graph.sum(dim = 0), -0.5))
        laplacian = torch.eye(graph.shape[0]).to(device) - torch.matmul(degree, torch.matmul(knn_graph, degree))
        values, eigs = torch.linalg.eigh(laplacian)

        print("finished computing eigenvalues")
        
        if dict_config['model'] == 'mlp' and dict_config['filtering'] is not None:
            x_train = torch.einsum("bid,ds->bis", x_train, eigs[:,dict_config['filtering']])
            x_val = torch.einsum("bid,ds->bis", x_val, eigs[:,dict_config['filtering']])
            x_test = torch.einsum("bid,ds->bis", x_test, eigs[:,dict_config['filtering']])
    if dict_config['model'] == 'mlp':
        graph = None
    dim = x_train.shape[-1]
    num_classes = int(torch.max(y_train).item()) + 1
    print("filtering :",dict_config['filtering'] is not None)
    print("dataset contains", num_classes, "classes and", dim, "dimensions")
    return bs, device,x_train,x_val,x_test,y_train,y_val,y_test,eigs,graph,dim,num_classes

###
            
class modelMLP(nn.Module):
    def __init__(self, depth, hidden_units,num_classes,dim):
        super(modelMLP, self).__init__()
        self.depth_filters = depth,hidden_units
        layers = []
        hidden_units = 2 ** hidden_units        
        last_dim = dim
        for i in range(depth):
            layers.append(nn.Linear(last_dim, hidden_units, bias = False))
            if i < depth - 1:
                layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU())
            last_dim = hidden_units
        layers = nn.ModuleList(layers)
        self.layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(last_dim, num_classes)
        self.name = 'modelMLP'
        self.mask = dim
        self.dim = dim
        

    def forward(self, x):
        if (type(self.mask) != int):
            if (len(self.mask) != self.dim):
                self.layers[0].weight.data = self.layers[0].weight*self.mask.to(device).float()
        x = x.reshape(x.shape[0], x.shape[-1])
        features = self.layers(x)
        out = self.final_layer(features)
        return out, features


    
class spectralLayer(nn.Module):
    def __init__(self, c_in, c_out, eigs, mask):
        super(spectralLayer, self).__init__()
        self.conv = nn.Conv1d(c_in, c_out, eigs[:,mask].shape[1], bias = False)
        self.mask = mask
        self.eigs = eigs

    def forward(self, x):
        spectral_x = torch.einsum("bid,ds->bis", x, self.eigs[:,self.mask])
        if self.conv.weight.shape[-1] !=  len(self.mask):
            conv_x = torch.einsum("bis,ois->bos", spectral_x, self.conv.weight[:,:,self.mask])
        else :
            conv_x = torch.einsum("bis,ois->bos", spectral_x, self.conv.weight)
        graph_x = torch.einsum("bos,ds->bod", conv_x, self.eigs[:,self.mask])
        return graph_x

class basicBlock(nn.Module):
    def __init__(self, filters_in, filters_out, eigs, mask):
        super(basicBlock, self).__init__()
        self.conv1 = spectralLayer(filters_in, filters_in,eigs=eigs, mask = mask)
        self.bn1 = nn.BatchNorm1d(filters_in)
        self.conv2 = spectralLayer(filters_in, filters_out,eigs=eigs, mask = mask)
        self.bn2 = nn.BatchNorm1d(filters_out)
        self.do = nn.Dropout(0.5)
        self.sc = False
        if filters_in != filters_out:
            self.sc = True
            self.convsc = nn.Conv1d(filters_in, filters_out, 1)
            self.bnsc = nn.BatchNorm1d(filters_out)

    def forward(self, x):
        x = self.do(x)
        out1 = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out1 = self.do(out1)
        out2 = torch.nn.functional.relu(self.bn2(self.conv2(out1)))
        out2 = self.do(out2)
        if self.sc:
            return self.bnsc(self.convsc(x)) + out2
        else:
            return x + out2

class modelGraph(nn.Module):
    def __init__(self, depth, filters, num_classes,eigs, mask):
        super(modelGraph, self).__init__()
        self.mask = mask
        try : 
            if self.mask == None:
                self.mask = np.arange(eigs.shape[1])
        except : 
            pass
        self.depth_filters = depth,filters
        filters = 2 ** filters
        self.embed = spectralLayer(1, filters, eigs, mask = self.mask)
        self.bn = nn.BatchNorm1d(filters)
        layers = [basicBlock(filters, filters, eigs, mask = self.mask) for i in range(depth)]        
        self.layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(filters, num_classes)
        self.name = 'modelGraph'
        
    def forward(self, x):
        embed = torch.nn.functional.relu(self.bn(self.embed(x)))
        pre_z = self.layers(embed)
        features = pre_z.mean(-1)
        out = self.final_layer(features)
        return out, features

def test(model, x_val, y_val,x_test, y_test, test = False):
    model.eval()
    (x, y) = (x_val, y_val) if not test else (x_test, y_test)
    with torch.no_grad():
        score, total = 0, 0
        for i in range(x.shape[0] // bs + 1 if x.shape[0] % bs != 0 else 0):
            data, target = x[bs*i: bs*(i+1)], y[bs*i: bs*(i+1)]
            output, _ = model(data)
            decisions = torch.argmax(output, dim = 1)
            score += (decisions == target).int().sum()
            total += decisions.shape[0]
    return (score / total).item()

def train_epoch(model,x_train,y_train, optimizer, mixup = False):
    model.train()
    data_perm = torch.randperm(x_train.shape[0])
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total_bs = 0
    for i in range(x_train.shape[0] // bs + (1 if x_train.shape[0] % bs != 0 else 0)):
        data, target = x_train[data_perm[bs*i: bs*(i+1)]], y_train[data_perm[bs*i: bs*(i+1)]]
        if mixup:
            alpha = random.random()
            perm = torch.randperm(data.shape[0])
            data = alpha * data + (1 - alpha) * data[perm]
        output, _ = model(data)
        if mixup:
            loss = alpha * criterion(output, target) + (1 - alpha) * criterion(output[perm], target)
        else:
            loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        total_bs += 1
        optimizer.step()
    return (total_loss / total_bs)

def train_epoch_pruning(model,x_train,y_train, optimizer,epoch,pruning_k,total_n_epoch, mixup = False,amin=1e-6,amax=1e4):
    model.train()
    data_perm = torch.randperm(x_train.shape[0])
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total_bs = 0
    nb_iter_per_epoch = np.ceil(len(x_train)/bs)
    sfinal = total_n_epoch*nb_iter_per_epoch
    s = epoch*nb_iter_per_epoch
    for i in range(x_train.shape[0] // bs + (1 if x_train.shape[0] % bs != 0 else 0)):
        data, target = x_train[data_perm[bs*i: bs*(i+1)]], y_train[data_perm[bs*i: bs*(i+1)]]
        if mixup:
            alpha = random.random()
            perm = torch.randperm(data.shape[0])
            data = alpha * data + (1 - alpha) * data[perm]
        output, _ = model(data)
        if mixup:
            loss = alpha * criterion(output, target) + (1 - alpha) * criterion(output[perm], target)
        else:
            loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        total_bs += 1
        s += 1
        apply_pruning(model,amin,amax,s,sfinal,device,pruning_k)
        optimizer.step()

    return (total_loss / total_bs)



def train(model,x_train,y_train,x_val, y_val,x_test, y_test, pruning=False,pruning_k =None,patience=5,amin=1e-6,amax=1e4,total_n_epoch = total_n_epoch_, silent = False):
    if model.name == 'modelMLP' and pruning:
        print('using sgd')
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, nesterov = True, weight_decay = 5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [30,60,90], gamma = 0.1)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=patience/2)
    else : 
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = None
    best_val, best_test,best_loss = 0, 0, 1e3 
    patience_counter= 0
    for epoch in range(total_n_epoch):
        if pruning : 
            loss = train_epoch_pruning(model,x_train,y_train,optimizer,epoch,pruning_k,total_n_epoch,True,amin,amax)
            loss = train_epoch_pruning(model,x_train,y_train,optimizer,epoch,pruning_k,total_n_epoch,False,amin,amax)
        else : 
            loss = train_epoch(model,x_train,y_train, optimizer, mixup = True)
            loss = train_epoch(model,x_train,y_train, optimizer)

        if not silent:
            print("\rEpoch {:3d} with loss {:.5f}".format(epoch, loss), end = ' ')
        val_acc = test(model,x_val, y_val,x_test, y_test, test = False)
        test_acc = test(model, x_val, y_val,x_test, y_test, test = True)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
        if not silent:
            print("val {:.3f} test {:.3f} best test {:.3f}".format(val_acc, test_acc, best_test), end = '')
        if scheduler:
            #scheduler.step(loss)
            scheduler.step()
        if patience :
            if round(loss,3) < round(best_loss,3):
                best_loss=loss
                patience_counter = 0
            else :
                patience_counter+=1
            if patience_counter==patience:
                break
    if pruning :
        mask_freq(model,pruning_k)
        if x_train.shape[0]<5000:
            best_test,best_val,loss,epoch = train(model,x_train,y_train,x_val, y_val,x_test, y_test, pruning=False,patience=25,total_n_epoch=200)
        else : 
            best_test,best_val,loss,epoch = train(model,x_train,y_train,x_val, y_val,x_test, y_test, pruning=False,patience=8,total_n_epoch=80)

    return best_test,best_val,loss,epoch


def count_parameters(model):
    print(np.sum([param.numel() for param in model.parameters()]))

def avg_score(dict_config,params,graph,num_classes,eigs,dim,x_train,y_train,x_val, y_val,x_test, y_test, n_runs = number_of_runs,use_wandb=False):
    #print(params)
    #if str(params) in dict_results.keys():
    #    return (dict_results[str(params)],),1,1
    #if dict_config['model']=='mlp':
    #    dict_config['param_0']=3
    #    dict_config['param_1']=10
    #    params[0] = 3
    #    params[1] = 10
    #else :
    #    dict_config['param_0']=4
    #    dict_config['param_1']=7
    #    params[0] = 4
    #    params[1] = 7
    #dict_config['param_0'] = params[0]
    #dict_config['param_1'] = params[1]
    if use_wandb:
        wandb.init(project="xxx", entity='yyy',config = dict_config,settings=wandb.Settings(start_method='fork'))
    avg_test_score = 0
    avg_val_score = 0
    avg_train_loss = 0
    test_scores = []
    val_scores = []
    epochs_max = []
    for i in range(n_runs):
        print("test", i+1)
        if graph is None :
            model = modelMLP(params[0],params[1],num_classes,dim).to(device)
        else : 
            model = modelGraph(params[0],params[1],num_classes,eigs,mask = dict_config['filtering']).to(device)
        best_test,best_val,train_loss,epoch_ = train(model,x_train,y_train,x_val, y_val,x_test, y_test,dict_config['pruning'],dict_config['pruning_k'],dict_config['patience'],dict_config['pruning_amin'],dict_config['pruning_amax'],dict_config['n_epoch'])
        avg_test_score += best_test
        avg_val_score += best_val
        avg_train_loss += train_loss
        test_scores.append(best_test)
        val_scores.append(best_val)
        epochs_max.append(epoch_)
    epochs_mean = np.mean(epochs_max)
    stats_test = stats(test_scores)
    stats_val = stats(val_scores)
    if use_wandb:
        wandb.log({'test_acc':avg_test_score/n_runs,'t_std':stats_test[1],'t_low':stats_test[2], 't_up':stats_test[3], 't_min':stats_test[4],'t_max':stats_test[5],
                    'val_acc':avg_val_score/n_runs,'v_std':stats_val[1],'v_low':stats_val[2], 'v_up':stats_val[3], 'v_min':stats_val[4],'v_max':stats_val[5],
                    'train_loss':avg_train_loss/n_runs,'epochs_mean':epochs_mean
        })
        wandb.finish()
    #dict_results[str(params)]=stats_test[0]
    return stats_test,stats_val, model
    
def search_params(params, best_score, index,dict_config,graph,num_classes,eigs,dim,x_train,y_train,x_val, y_val,x_test, y_test):
    if index == len(params):
        print("finished")
        exit()
        #return None
    print("Testing with params", params, "best score is", best_score)
    params[index] += 1
    s_t,_,_ = avg_score(dict_config,params,graph,num_classes,eigs,dim,x_train,y_train,x_val, y_val,x_test, y_test,n_runs=number_of_runs)
    score = s_t[0]
    if score > best_score:
        search_params(params, score, 0,dict_config,graph,num_classes,eigs,dim,x_train,y_train,x_val, y_val,x_test, y_test)
    params[index] -= 2
    if params[index] >= 0:
        s_t,_,_ = avg_score(dict_config,params,graph,num_classes,eigs,dim,x_train,y_train,x_val, y_val,x_test, y_test,n_runs=number_of_runs)
        score = s_t[0]
        if score > best_score:
            search_params(params, score, 0,dict_config,graph,num_classes,eigs,dim,x_train,y_train,x_val, y_val,x_test, y_test)
    params[index] += 1
    search_params(params, best_score, index + 1,dict_config,graph,num_classes,eigs,dim,x_train,y_train,x_val, y_val,x_test, y_test)


def stats(scores, name=" "):
    if len(scores) == 1:
        low, up = 0., 1.
    elif len(scores) < 30:
        low, up = st.t.interval(0.95, df = len(scores) - 1, loc = np.mean(scores), scale = st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc = np.mean(scores), scale = st.sem(scores))
    #if name == "":
    #    return np.mean(scores), up - np.mean(scores)
    mean_ = np.mean(scores)
    std_ = np.std(scores)
    min_ = np.min(scores)
    max_ = np.max(scores)
    print("{:s} {:.2f} (± {:.2f}) (conf: [{:.2f}, {:.2f}]) (worst: {:.2f}, best: {:.2f})".format(name, 100 * mean_ , 100 * std_, 100 * low, 100 * up, 100 * min_, 100 * max_))
    return mean_,std_,low,up,min_,max_
        

### pruning ###

def compute_swd(amin,amax,s,sfinal):
    return amin*(amax/amin)**(s/sfinal)


def compute_index_to_prune(model,k):
    if model.name == 'modelMLP':
        linear_weights = model.layers[0].weight.detach().to('cpu')
        _, index_to_prune = torch.sort(torch.norm(linear_weights,dim=0))
        return index_to_prune[:k]
    else : 
        conv_weights = []
        filters = 2**model.depth_filters[1]
        for n,p in model.named_parameters():
            if 'conv.weight' in n:
                if 'embed' in n :
                    conv0 = torch.zeros([filters,filters,360])
                    conv0[0] = p.squeeze(1)
                    conv_weights.append(conv0)
                else : 
                    conv_weights.append(p.detach().to('cpu'))
        conv_weights = torch.stack(conv_weights).detach().to('cpu')
        _, index_to_prune = torch.sort(torch.linalg.norm(conv_weights,dim=[0,1,2]))
        return index_to_prune[:k]

def apply_pruning(model,amin,amax,s,sfinal,device,k):
    index_to_prune = compute_index_to_prune(model,k)
    swd = compute_swd(amin,amax,s,sfinal)
    if model.name == 'modelMLP':
        mask = torch.zeros(model.layers[0].weight.shape).to(device)
        mask[:,index_to_prune]=1
        model.layers[0].weight.grad += swd * model.layers[0].weight *mask
    else : 
        for n,p in model.named_parameters():
            if 'conv.weight' in n:
                mask = torch.zeros(p.shape).to(device)
                mask[:,:,index_to_prune]=1
                p.grad += swd * p.data *mask#.float()
    #return np.mean(index_to_prune.numpy())
            

            
def mask_freq(model,k):
    if model.name =='modelMLP':
        index_to_prune = compute_index_to_prune(model,k)
        index_to_keep = [x for x in np.arange(360) if x not in index_to_prune]
        mask = torch.zeros(model.layers[0].weight.shape)
        mask[:,index_to_keep] = 1
        model.mask = mask
        # model.layers[0].weight*mask.to(device).float()
        #model.layers[0].weight = torch.nn.Parameter(model.layers[0].weight*mask.to(device).float())
    else : 
        model.mask = [i for i in model.mask if i not in compute_index_to_prune(model,k)]
        model.embed.mask = model.mask
        #model.embed.conv.weight = model.embed.conv.weight[:,:,model.mask]
        for i in range(model.depth_filters[0]):
            model.layers[i].conv1.mask = model.mask
            model.layers[i].conv2.mask = model.mask

def correct_dict(dict_config):
    #if dict_config['model'] == 'mlp':
    #    dict_config['graph'] = None
    if dict_config['pruning'] == False:
        dict_config['pruning_k']= None
        if dict_config['dataset'] == 'IBC':
            dict_config['patience'] = 15
            dict_config['n_epoch']= 150
        if dict_config['dataset'] == 'HCP':
            dict_config['patience'] = 8
            dict_config['n_epoch']= 90
        dict_config['pruning_amin']=None		
        dict_config['pruning_amax']=None
    else : 
        dict_config['patience']= None
        #dict_config['amin']=
        if dict_config['pruning_k']==None:
            print('WARNING !!!!!!! specify pruning k')
        if dict_config['model'] == 'mlp':
            dict_config['n_epoch']=130
        if dict_config['model'] != 'mlp':
            if dict_config['dataset'] == 'IBC':
                dict_config['n_epoch']  = 140
                dict_config['patience'] = None
            if dict_config['dataset'] == 'HCP':
                dict_config['n_epoch']  = 65
                dict_config['patience'] = None
    if (dict_config['model'] == 'mlp') and (dict_config['pruning']==True):
        dict_config['filtering'] = np.arange(360)
    if (dict_config['model'] != 'mlp') and (dict_config['pruning']==True):
        dict_config['filtering'] = None
    if dict_config['n_epoch'] is None :
        dict_config['n_epoch']=total_n_epoch_

    return dict_config