import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel, BertTokenizer, BertModel

from config import get_config
from dataloader import data_loader, load_dataset
from model import Lstm_Model, BiLstm_Model, Transformer_CNN_RNN


class ZhiYinBoy:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        
        self.logger.info('> creating model {}'.format(args.model_name))
        
        # # Create model
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.input_size = 768
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('roberta-base')
        elif args.model_name =='chinese_bert':      
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('bert-base-chinese')
        elif args.model_name =='chinese_roberta':
            self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
            self.input_size = 768
            base_model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        else:
            raise ValueError('unknown model')
        # # Operate the method
        if args.method_name == 'lstm':
            self.Mymodel = Lstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'bilstm':
            self.Mymodel = BiLstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'lstm+textcnn':
            self.Mymodel = Transformer_CNN_RNN(base_model, args.num_classes)
        else:
            raise ValueError('unknown method')
        # self.Mymodel = Lstm_Model(base_model, args.num_classes,self.input_size)

        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        # Turn on the train mode
        self.Mymodel.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            predicts = self.Mymodel(inputs)
            loss = criterion(predicts, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)

        return train_loss / n_train, n_correct / n_train


    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        # Turn on the eval mode
        self.Mymodel.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)

        return test_loss / n_test, n_correct / n_test

    def run(self):
        if self.args.test_file == self.args.train_file:
            self.logger.info('run in the same dataset, auto split')
            train_dataloader, test_dataloader = load_dataset(tokenizer=self.tokenizer,
                                                             train_batch_size=self.args.train_batch_size,
                                                             test_batch_size=self.args.test_batch_size,
                                                             model_name=self.args.model_name,
                                                             method_name=self.args.method_name,
                                                             workers=self.args.workers,
                                                             data_file='DataSet\\' + self.args.test_file)
        else:
            train_dataloader, _ = data_loader(tokenizer=self.tokenizer,
                                              batch_size=self.args.train_batch_size,
                                              model_name=self.args.model_name,
                                              method_name=self.args.method_name,
                                              workers=self.args.workers,
                                              data_file='DataSet\\' + self.args.train_file,
                                              ratio=0.1)
            test_dataloader, _ = data_loader(tokenizer=self.tokenizer,
                                             batch_size=self.args.test_batch_size,
                                             model_name=self.args.model_name,
                                             method_name=self.args.method_name,
                                             workers=self.args.workers,
                                             data_file='DataSet\\' + self.args.test_file,
                                             ratio=0.05)

        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        l_acc, l_trloss, l_teloss, l_epo = [], [], [], []
        # Get the best_loss and the best_acc
        best_loss, best_acc = 0, 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc = self._test(test_dataloader, criterion)
            l_epo.append(epoch), l_acc.append(test_acc), l_trloss.append(train_loss), l_teloss.append(test_loss)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc * 100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc * 100))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        path = f'Model\\{args.language}+{args.model_type}+{args.model_name}+{args.method_name}'
        
        torch.save(self.Mymodel, path+'.pt')


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    nb = ZhiYinBoy(args, logger)
    nb.run()
