# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
### Helper Functions and global settings
FORCE_CPU = False
line_length = 60
def pretty_print_divider(n=1, lb_n=0, char="#"):
    if isinstance(n, bool):
        n = 1 if n else 0
    if lb_n > 0:
        print("\n" * lb_n, end="")
    elif n > 1:
        print()
    for _ in range(n):
        print(char * line_length)

def pretty_print(message, pb=False, pa=False, lb_n=0, char="#"):
    pretty_print_divider(pb, lb_n=lb_n, char=char)
    available_space = line_length - 7
    formatted_message = f"{char * 3} {message}"
    padding = line_length - len(formatted_message) - 4
    if padding < 0:
        formatted_message = formatted_message[:line_length-7] + "..."
        padding = 0
    print(formatted_message + " " * padding + f" {char * 3}")
    pretty_print_divider(pa, char=char)
# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

torch.manual_seed(8) # for reproduce
# Dynamically set the device to use (either GPU or CPU)
if torch.cuda.is_available() and not FORCE_CPU:
    device_count = torch.cuda.device_count()
    if device_count > 1:
        pretty_print(f"multiple GPUs detected ({device_count})", pb=True)
        best_device_id = -1
        max_free_mem = 0
        for i in range(device_count):
            free, _ = torch.cuda.mem_get_info(i)
            pretty_print(f"GPU {i} free memory: {free}")
            if free > max_free_mem:
                max_free_mem = free
                best_device_id = i
        device = torch.device(f'cuda:{best_device_id}')
        out_device = f"GPU: cuda:{best_device_id}"
    else:
        device = torch.device('cuda')
        out_device = "GPU: cuda:0"
else:
    device = torch.device('cpu')
    out_device = "CPU"
torch.backends.cudnn.benchmark = True
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)
torch.nn.Module.dump_patches = True

pretty_print(f"Using device: {out_device}", pb=True, pa=True)


# %%
import time
import numpy as np
import gc
import sys
sys.setrecursionlimit(50000)
import pickle

# from tensorboardX import SummaryWriter

import copy
import pandas as pd
#then import my own modules
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight, featurize_smiles_from_dict


# %%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score


# %%
# from rdkit.Chem import rdMolDescriptors, MolSurf
# from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
# from rdkit.Chem import AllChem
from rdkit.Chem import QED
# get_ipython().run_line_magic('matplotlib', 'inline')
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from IPython.display import SVG, display
import seaborn as sns; sns.set(color_codes=True)


# %%
task_name = 'BBBP'
tasks = ['BBBP']
raw_filename = "../data/BBBP.csv"
feature_filename = raw_filename.replace('.csv','.pickle')
filename = raw_filename.replace('.csv','')
prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
smiles_tasks_df = pd.read_csv(raw_filename)
smilesList = smiles_tasks_df.smiles.values
pretty_print(f"number of all smiles: {len(smilesList)}")
atom_num_dist = []
remained_smiles = []
canonical_smiles_list = []
for smiles in smilesList:
    try:        
        mol = Chem.MolFromSmiles(smiles)
        atom_num_dist.append(len(mol.GetAtoms()))
        remained_smiles.append(smiles)
        canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    except:
        pretty_print(f"not successfully processed smiles: {smiles}")
        pass
pretty_print(f"number of successfully processed smiles: {len(remained_smiles)}")
smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
# print(smiles_tasks_df)
smiles_tasks_df['cano_smiles'] =canonical_smiles_list
assert canonical_smiles_list[8]==Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'][8]), isomericSmiles=True)

plt.figure(figsize=(5, 3))
sns.set(font_scale=1.5)
ax = sns.distplot(atom_num_dist, bins=28, kde=False)
plt.tight_layout()
# plt.savefig("atom_num_dist_"+prefix_filename+".png",dpi=200)
plt.show()
plt.close()

# print(len([i for i in atom_num_dist if i<51]),len([i for i in atom_num_dist if i>50]))


# %%
random_seed = 188
random_seed = int(time.time())
start_time = str(time.ctime()).replace(':','-').replace(' ','_')
start = time.time()

batch_size = 100
epochs = 800
p_dropout = 0.1
fingerprint_dim = 150

radius = 3
T = 2
weight_decay = 2.9 # also known as l2_regularization_lambda
learning_rate = 3.5
per_task_output_units_num = 2 # for classification model with 2 classes
output_units_num = len(tasks) * per_task_output_units_num


# %%
smilesList = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms())<101]
uncovered = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms())>100]

smiles_tasks_df = smiles_tasks_df[~smiles_tasks_df["cano_smiles"].isin(uncovered)]

if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb" ))
else:
    feature_dicts = save_smiles_dicts(smilesList,filename)
# feature_dicts = get_smiles_dicts(smilesList)

remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
uncovered_df = smiles_tasks_df.drop(remained_df.index)
uncovered_df


# %%
weights = []
for i,task in enumerate(tasks):    
    negative_df = remained_df[remained_df[task] == 0][["smiles",task]]
    positive_df = remained_df[remained_df[task] == 1][["smiles",task]]
    weights.append([(positive_df.shape[0]+negative_df.shape[0])/negative_df.shape[0],\
                    (positive_df.shape[0]+negative_df.shape[0])/positive_df.shape[0]])

test_df = remained_df.sample(frac=1/10, random_state=random_seed) # test set
training_data = remained_df.drop(test_df.index) # training data

# training data is further divided into validation set and train set
valid_df = training_data.sample(frac=1/9, random_state=random_seed) # validation set
train_df = training_data.drop(valid_df.index) # train set
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

valid_df.to_csv('sets/valid_df.csv')
train_df.to_csv('sets/train_df.csv')
test_df.to_csv('sets/test_df.csv')


# %%
x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smilesList[0]],feature_dicts)
num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]

loss_function = [nn.CrossEntropyLoss(torch.tensor(weight),reduction='mean') for weight in weights]
model = Fingerprint(radius, T, num_atom_features,num_bond_features,
            fingerprint_dim, output_units_num, p_dropout)
model.to(device)
# tensorboard = SummaryWriter(log_dir="runs/"+start_time+"_"+prefix_filename+"_"+str(fingerprint_dim)+"_"+str(p_dropout))

# optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
pretty_print(f"Number of parameters: {params}")
for name, param in model.named_parameters():
    if param.requires_grad:
        pretty_print(f"{name} {param.data.shape}")



# %%
def train(model, dataset, optimizer, loss_function):
    model.train()
    np.random.seed(epoch)
    valList = np.arange(0,dataset.shape[0])
    #shuffle them
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch,:]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.tensor(x_atom),
                                                 torch.tensor(x_bonds),
                                                 torch.tensor(x_atom_index, dtype=torch.long),
                                                 torch.tensor(x_bond_index, dtype=torch.long),
                                                 torch.tensor(x_mask))
#         print(torch.Tensor(x_atom).size(),torch.Tensor(x_bonds).size(),torch.cuda.LongTensor(x_atom_index).size(),torch.cuda.LongTensor(x_bond_index).size(),torch.Tensor(x_mask).size())

        model.zero_grad()
        # Step 4. Compute your loss function. (Again, Torch wants the target wrapped in a variable)
        loss = 0.0
        for i,task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                    per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val==0) | (y_val==1))[0]
#             validInds = np.where(y_val != -1)[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.tensor(validInds, dtype=torch.long).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)

            loss += loss_function[i](
                y_pred_adjust,
                torch.tensor(y_val_adjust, dtype=torch.long))
        # Step 5. Do the backward pass and update the gradient
#             print(y_val,y_pred,validInds,y_val_adjust,y_pred_adjust)
        loss.backward()
        optimizer.step()
def eval(model, dataset):
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    losses_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   
    for counter, test_batch in enumerate(batch_list):
        batch_df = dataset.loc[test_batch,:]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.tensor(x_atom),
                                                 torch.tensor(x_bonds),
                                                 torch.tensor(x_atom_index, dtype=torch.long),
                                                 torch.tensor(x_bond_index, dtype=torch.long),
                                                 torch.tensor(x_mask))
        atom_pred = atoms_prediction.data[:,:,1].unsqueeze(2).cpu().numpy()
        for i,task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                    per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val==0) | (y_val==1))[0]
#             validInds = np.where((y_val=='0') | (y_val=='1'))[0]
#             print(validInds)
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.tensor(validInds, dtype=torch.long).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
#             print(validInds)
            loss = loss_function[i](
                y_pred_adjust,
                torch.tensor(y_val_adjust, dtype=torch.long))
#             print(y_pred_adjust)
            y_pred_adjust = F.softmax(y_pred_adjust,dim=-1).data.cpu().numpy()[:,1]
            losses_list.append(loss.cpu().detach().numpy())
            try:
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
            except:
                y_val_list[i] = []
                y_pred_list[i] = []
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
#             print(y_val,y_pred,validInds,y_val_adjust,y_pred_adjust)            
    test_roc = [roc_auc_score(y_val_list[i], y_pred_list[i]) for i in range(len(tasks))]
    test_prc = [auc(precision_recall_curve(y_val_list[i], y_pred_list[i])[1],precision_recall_curve(y_val_list[i], y_pred_list[i])[0]) for i in range(len(tasks))]
#     test_prc = auc(recall, precision)
    test_precision = [precision_score(y_val_list[i],
                                     (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    test_recall = [recall_score(y_val_list[i],
                               (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    test_loss = np.array(losses_list).mean()

    return test_roc, test_prc, test_precision, test_recall, test_loss


# %%
# best_param ={}
# best_param["roc_epoch"] = 0
# best_param["loss_epoch"] = 0
# best_param["valid_roc"] = 0
# best_param["valid_loss"] = 9e8

# for epoch in range(epochs):    
#     train_roc, train_prc, train_precision, train_recall, train_loss = eval(model, train_df)
#     valid_roc, valid_prc, valid_precision, valid_recall, valid_loss = eval(model, valid_df)
#     train_roc_mean = np.array(train_roc).mean()
#     valid_roc_mean = np.array(valid_roc).mean()

# #     tensorboard.add_scalars('ROC',{'train_roc':train_roc_mean,'valid_roc':valid_roc_mean},epoch)
# #     tensorboard.add_scalars('Losses',{'train_losses':train_loss,'valid_losses':valid_loss},epoch)

#     if valid_roc_mean > best_param["valid_roc"]:
#         best_param["roc_epoch"] = epoch
#         best_param["valid_roc"] = valid_roc_mean
#         if valid_roc_mean > 0.87:
#              torch.save(model, 'saved_models/model_'+prefix_filename+'_'+start_time+'_'+str(epoch)+'.pt')             
#     if valid_loss < best_param["valid_loss"]:
#         best_param["loss_epoch"] = epoch
#         best_param["valid_loss"] = valid_loss

#     pretty_print(f"EPOCH: {epoch}", pb=3 if epoch == 0 else False)
#     pretty_print(f"train_roc: {train_roc}")
#     pretty_print(f"valid_roc: {valid_roc}", pa=True)

#     if (epoch - best_param["roc_epoch"] >18) and (epoch - best_param["loss_epoch"] >28):        
#         break

#     train(model, train_df, optimizer, loss_function)


# %%
# # evaluate model
# best_model = torch.load('saved_models/model_'+prefix_filename+'_'+start_time+'_'+str(best_param["roc_epoch"])+'.pt')     

# # best_model_dict = best_model.state_dict()
# # best_model_wts = copy.deepcopy(best_model_dict)

# # model.load_state_dict(best_model_wts)
# # (best_model.align[0].weight == model.align[0].weight).all()

# test_roc, test_prc, test_precision, test_recall, test_losses = eval(best_model, test_df)

# pretty_print(f"best epoch: {best_param['roc_epoch']}", pb=True)
# pretty_print(f"test_roc: {test_roc}")
# pretty_print(f"test_roc_mean: {np.array(test_roc).mean()}", pa=True)

# %%
# Inference on a single SMILES string
model_filepath = 'saved_models/model_BBBP_Mon_Jul__7_23-17-57_2025_327.pt'
smile_to_test = 'O=C(C)Oc1ccccc1C(=O)O' # Aspirin

if os.path.isfile(model_filepath):
    pretty_print(f"Loading model from {model_filepath}", pb=True)
    model = torch.load(model_filepath, map_location=device, weights_only=False)
    model.eval()

    pretty_print(f"Running inference for SMILES: {smile_to_test}")

    try:
        # Featurize the SMILES string using the new function
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = featurize_smiles_from_dict(smile_to_test, feature_dicts)

        # Convert numpy arrays to tensors
        x_atom_tensor = torch.tensor(x_atom)
        x_bonds_tensor = torch.tensor(x_bonds)
        x_atom_index_tensor = torch.tensor(x_atom_index, dtype=torch.long)
        x_bond_index_tensor = torch.tensor(x_bond_index, dtype=torch.long)
        x_mask_tensor = torch.tensor(x_mask)
        
        # Perform prediction
        with torch.no_grad():
            _, mol_prediction = model(x_atom_tensor, x_bonds_tensor, x_atom_index_tensor, x_bond_index_tensor, x_mask_tensor)

        # Process the output
        probabilities = F.softmax(mol_prediction, dim=1)
        prob_class_0 = probabilities[0, 0].item()
        prob_class_1 = probabilities[0, 1].item()
        predicted_class = torch.argmax(probabilities, dim=1).item()

        pretty_print(f"Raw model output (logits): {mol_prediction.cpu().numpy().flatten()}", pa=True)

        # BBBP classification (1: Penetrates, 0: Doesn't Penetrate)
        
        pretty_print(f"Prediction for SMILES: {smile_to_test}", pb=True)
        pretty_print(f"Probability of NOT crossing BBB (class 0): {prob_class_0:.4f}")
        pretty_print(f"Probability of crossing BBB (class 1): {prob_class_1:.4f}")
        pretty_print(f"Predicted class: {predicted_class} ({'Crosses BBB' if predicted_class == 1 else 'Does not cross BBB'})", pa=True)

    except ValueError as e:
        print(f"Error featurizing SMILES: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


else:
    print(f"Model file not found at: {model_filepath}")