import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import logging, AutoTokenizer, BertTokenizer

from config import get_config
from dataloader import data_loader


def show_predict(coms, pres, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write('<weibos>\n')
        length = len(coms)
        for i in range(0, length):
            if pres[i] == 0:
                pre = -1
            else:
                pre = 1
            f.write(f'\t<weibo id="{i+1}" polarity="{pre}">' + coms[i] + '</weibo>\n')
        f.write('</weibos>\n')


class Tester:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.logger.info(f'> Testing for {args.test_file} using {args.model_name} {args.method_name}')

        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
        elif args.model_name =='chinese_bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', add_prefix_space=True)
        elif args.model_name =='chinese_roberta':
            self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        else:
            raise ValueError('unknown model')

        self.Mymodel = torch.load(f'Model\\{args.language}+{args.model_type}+{args.model_name}+{args.method_name}.pt')

    def run(self):
        dataloader, comments = data_loader(tokenizer=self.tokenizer,
                                           batch_size=self.args.test_batch_size,
                                           model_name=self.args.model_name,
                                           method_name=self.args.model_name,
                                           workers=self.args.workers,
                                           data_file='DataSet\\' + self.args.test_file,
                                           ratio=1.0,
                                           seq=True)
        criterion = nn.CrossEntropyLoss()

        test_loss, n_correct, n_test = 0, 0, 0
        self.Mymodel.eval()
        pre_list = []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                pres = torch.argmax(predicts, dim=1)
                pre_list += pres.tolist()
                n_correct += (pres == targets).sum().item()
                n_test += targets.size(0)

        test_loss, test_acc = test_loss / n_test, n_correct / n_test
        if ('sample' not in self.args.test_file) and ('input' not in self.args.test_file):
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc * 100))
        show_predict(comments, pre_list,
                     f'Output\\results_{args.language}+{args.model_type}+{args.model_name}+{args.method_name}.xml')
        self.logger.info('> Predict has completed')
        self.logger.info(f'> The predicts has been saved in ' +
                         f'results_{args.language}+{args.model_type}+{args.model_name}+{args.method_name}.xml')


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    nb = Tester(args, logger)
    nb.run()
