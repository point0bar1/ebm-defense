####################################################################
# ## NATURAL AND ADVERSARIAL TRAINING FOR WIDERESNET CLASSIFIER ## #
####################################################################

import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader

import json
import datetime

from nets import WideResNet, conv_init
from utils import setup_exp, import_data

# json file with experiment config
CONFIG_FILE = './config_train_clf/cifar10_nat.json'


###############
# ## SETUP ## #
###############

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)
# directory for experiment results
exp_dir = config['exp_dir'] + '_' + datetime.datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p') + '_/'
# setup folders, save code, set seed and get device
setup_exp(exp_dir, config['seed'], ['checkpoints'], ['train_clf.py', 'nets.py', 'utils.py', CONFIG_FILE])

print('Processing data...')
# import train and test datasets and set up data loaders
train_data, num_classes = import_data(config['data_type'], True, True)
train_data_loader = DataLoader(train_data, config['batch_size'], shuffle=True, num_workers=config['num_workers'])
test_data = import_data(config['data_type'], False, False)[0]
test_data_loader = DataLoader(test_data, config['batch_size'], shuffle=False, num_workers=config['num_workers'])

print('Setting up network and optimizer...')
# network structure and weight init
clf = WideResNet(num_classes=num_classes).cuda()
clf.apply(conv_init)
# initialize optim
assert len(config['lr_list']) == len(config['lr_schedule']), 'lr_list and lr_schedule must have the same length'
optim = t.optim.SGD(clf.parameters(), config['lr_list'][0], config['momentum'], weight_decay=config['weight_decay'])

# loss criterion for logits
criterion = nn.CrossEntropyLoss()

# rescale adversarial parameters for attacks on images with pixel intensities in the range [-1, 1]
config['adv_eps'] *= 2.0 / 255.0
config['adv_eta'] *= 2.0 / 255.0


###############################################
# ## FUNCTIONS FOR ATTACK, TRAIN, AND TEST ## #
###############################################

# l_inf pgd attack
def attack(X, y, adv_steps):
    min_mask = t.clamp(X - config['adv_eps'], min=-1.0, max=1.0)
    max_mask = t.clamp(X + config['adv_eps'], min=-1.0, max=1.0)
    # random initialization in epsilon ball
    X_adv = t.clamp(X + config['adv_eps']*(2*t.rand_like(X)-1), min=-1.0, max=1.0)
    X_adv = t.autograd.Variable(X_adv, requires_grad=True)
    for step in range(adv_steps):
        # l_infinity attack on images by climbing loss within epsilon-ball
        attack_grad = t.autograd.grad(criterion(clf(X_adv), y), [X_adv])[0]
        X_adv.data += config['adv_eta'] * t.sign(attack_grad)
        X_adv.data = t.min(max_mask, other=t.max(min_mask, other=X_adv.data))
    return X_adv.detach()

# train model for single epoch through data
def train(epoch):
    clf.train()
    train_loss = 0
    correct = 0
    total = 0

    # update learning rate
    if (epoch + 1) in config['lr_schedule']:
        lr_ind = config['lr_schedule'].index(epoch + 1)
        for lr_gp in optim.param_groups:
            lr_gp['lr'] = config['lr_list'][lr_ind]

    for batch, (X_batch, y_batch) in enumerate(train_data_loader):
        X_train, y_train = X_batch.clone().cuda(), y_batch.cuda()

        if config['adv_steps_train'] > 0 and (epoch+1) >= config['adv_train_start']:
            # adversarial attacks on input images
            X_train = attack(X_train, y_train, config['adv_steps_train'])

        # logits for prediction and loss for weight update
        logits = clf(X_train)
        loss = criterion(logits, y_train)
        # update classifier weights
        optim.zero_grad()
        loss.backward()
        optim.step()

        # record batch info
        train_loss += loss.item()
        _, y_pred = t.max(logits.detach(), 1)
        correct += t.eq(y_pred, y_train).sum().cpu()
        total += y_train.nelement()

    # get and print train accuracy
    train_acc = 100 * float(correct) / float(total)
    print('Epoch {}: Train Loss={}   Train Acc={}%'.format(epoch+1, train_loss / (batch+1), train_acc))

# test model on withheld data
def test(epoch):
    clf.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch, (X_batch, y_batch) in enumerate(test_data_loader):
        X_test, y_test = X_batch.clone().cuda(), y_batch.cuda()

        if config['adv_steps_test'] > 0:
            # attack images
            X_test = attack(X_test, y_test, config['adv_steps_test'])

        # check test images
        with t.no_grad():
            logits = clf(X_test)
            loss = criterion(logits, y_test)

            # record batch info
            test_loss += loss.item()
            _, y_pred = t.max(logits, 1)
            correct += t.eq(y_pred, y_test).sum().cpu()
            total += y_test.nelement()

    # get and print test accuracy
    test_acc = 100 * float(correct) / float(total)
    print('Epoch {}: Test Loss={}   Test Acc={}%'.format(epoch+1, test_loss / (batch+1), test_acc))


#######################
# ## LEARNING LOOP # ##
#######################

print('Training has begun.')
for epoch in range(config['num_epochs']):
    train(epoch)
    if (epoch+1) % config['test_and_log_freq'] == 0:
        # save checkpoint
        t.save(clf.state_dict(), exp_dir + 'checkpoints/clf_' + str(epoch + 1) + '.pth')
        # save optim
        t.save(optim.state_dict(), exp_dir + 'checkpoints/optim.pth')
        # evaluate test data
        test(epoch)
