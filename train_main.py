from ABSA_CL import ABSA_CL
from pytorch_transformers import BertTokenizer

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy as np
from tqdm import tqdm, trange
import torch
import json

from data_utils import DataProcessor,seq_reduce_ret_len
from sklearn import metrics

from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler, PolyWarmUpScheduler
from apex import amp
from torchtext.data.metrics import bleu_score

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

class Instructor:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.dataset_list = args.dataset.split(",")
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, never_split=["<e1>", "</e1>"], do_basic_tokenize=True)
        self.data_processor = DataProcessor(args.data_dir, self.tokenizer,
                                            max_seq_len=args.max_seq_len,
                                            batch_size=args.batch_size)

        if args.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=args.device.index)))

    def saving_model(self, saving_model_path, model, optimizer):
        if not os.path.exists(saving_model_path):
            os.mkdir(saving_model_path)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(saving_model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(saving_model_path, CONFIG_NAME)
        output_optimizer_file = os.path.join(saving_model_path, "optimizer.pt")
        torch.save(model_to_save.state_dict(), output_model_file)
        with open(output_config_file, "w", encoding='utf-8') as writer:
            writer.write(model_to_save.config.to_json_string())
        torch.save({'optimizer': optimizer.state_dict(),
                    'master params': list(amp.master_params(optimizer))},
                   output_optimizer_file)

    def load_model(self, model, optimizer, saving_model_path):
        output_model_file = os.path.join(saving_model_path, WEIGHTS_NAME)
        output_optimizer_file = os.path.join(saving_model_path, "optimizer.pt")
        #model
        checkpoint_model = torch.load(output_model_file, map_location="cpu")
        model.load_state_dict(checkpoint_model)
        #optimizer
        checkpoint_optimizer = torch.load(output_optimizer_file, map_location="cpu")
        if self.args.fp16:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint_optimizer['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint_optimizer['master params']):
                param.data.copy_(saved_param.data)
        else:
            optimizer.load_state_dict(checkpoint_optimizer["optimizer"])
        return model, optimizer

    def save_args(self):
        output_args_file = os.path.join(self.args.outdir, 'training_args.bin')
        torch.save(self.args, output_args_file)

    def _evaluate_acc_f1(self, model, data_loader):
        model.eval()
        n_correct, n_total, loss_total, nb_tr_steps = 0, 0, 0, 0
        t_targets_all, t_outputs_all = None, None
        criteria_index = 0
        s_n_correct, s_n_total = 0, 0
        gold_sentences = []
        generated_sentences = []

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                reduce_seq_len = seq_reduce_ret_len(sample_batched["input_ids"])
                input_ids = sample_batched["input_ids"][:,:reduce_seq_len].to(self.device)
                aspect_mask = sample_batched["aspect_mask"][:,:reduce_seq_len].to(self.device)
                segment_ids = sample_batched["segment_ids"][:,:reduce_seq_len].to(self.device)
                attention_mask = sample_batched["input_mask"][:,:reduce_seq_len].to(self.device)
                label_ids = sample_batched["label_ids"].to(self.args.device)

                tag_seq, generated_seq = model(input_ids=input_ids, criteria_index=criteria_index, token_type_ids=segment_ids, aspect_mask=aspect_mask)

                nb_tr_steps += 1
                n_correct += (tag_seq == label_ids).sum().item()
                n_total += len(tag_seq)

                if t_targets_all is None:
                    t_targets_all = label_ids
                    t_outputs_all = tag_seq
                else:
                    t_targets_all = torch.cat((t_targets_all, label_ids), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, tag_seq), dim=0)

                for j in range(len(input_ids)):
                    temp_input_ids = input_ids[j].cpu().tolist()
                    gold_sentence = self.tokenizer.convert_ids_to_tokens(temp_input_ids)
                    index_sep = gold_sentence.index('[SEP]')
                    gold_sentence = gold_sentence[:index_sep+1]
                    gold_sentences.append([gold_sentence])
                    
        logger.info("nb_tr_examples: {}, nb_tr_steps: {}".format(n_total, nb_tr_steps))

        acc = n_correct / n_total
        
        f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1, 2], average='macro')
               
        return {
            "precision": acc,
            "f1": f1
        }

    def run(self):
        self.save_args()
        args = self.args
        num_labels = self.data_processor.get_tag_size()
        model = ABSA_CL(tag_size=3, bert_path=args.bert_model, embedding="bert", adv_coefficient=0.06, pooling=args.pooling, encoder=args.encoder, adversary=args.adversary, num_layers=args.num_layers)
        model = model.to(self.args.device)

        logger.info("Loading data...")
        train_dataloader, test_dataloader, dev_dataloader = self.data_processor.get_dataloader(self.dataset)
        logger.info("Loading finished.")
        num_train_optimization_steps = int(
            len(train_dataloader) / self.args.gradient_accumulation_steps) * self.args.num_epoch

        logger.info("trainset: {}, batch_size: {}, gradient_accumulation_steps: {}, num_epoch: {}, num_train_optimization_steps: {}".format(
            len(train_dataloader) * self.args.batch_size, self.args.batch_size, self.args.gradient_accumulation_steps,
            self.args.num_epoch, num_train_optimization_steps))

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        print("Number of parameters:", sum(p[1].numel() for p in param_optimizer if p[1].requires_grad))
        if self.args.fp16:
            logger.info("using fp16")
            try:
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=self.args.learning_rate,
                                  bias_correction=False)

            if self.args.loss_scale == 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale="dynamic")
            else:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale=self.args.loss_scale)
            scheduler = LinearWarmUpScheduler(optimizer, warmup=self.args.warmup_proportion,
                                              total_steps=num_train_optimization_steps)
        else:
            logger.info("using fp32")
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=self.args.learning_rate,
                                 warmup=self.args.warmup_proportion,
                                 t_total=num_train_optimization_steps)
            scheduler = None
        logger.info("Optimizer done...")

        logger.info("Training begins...")
        loss_ = self.train(model, optimizer, scheduler, train_dataloader, dev_dataloader, test_dataloader)


    def train_multi_criteria(self, model, optimizer, scheduler, train_data_loader, global_step):
        args = self.args
        tr_loss = 0
        optimizer.zero_grad()
        nb_tr_examples, nb_tr_steps = 0, 0
        criteria_index = 1
        for step, sample_batched in enumerate(tqdm(train_data_loader, desc="Iteration")):
            if args.max_steps > 0 and global_step > args.max_steps:
                break
            
            reduce_seq_len = seq_reduce_ret_len(sample_batched["input_ids"])
            input_ids = sample_batched["input_ids"][:,:reduce_seq_len].to(self.device)
            aspect_mask = sample_batched["aspect_mask"][:,:reduce_seq_len].to(self.device)
            segment_ids = sample_batched["segment_ids"][:,:reduce_seq_len].to(self.device)
            attention_mask = sample_batched["input_mask"][:,:reduce_seq_len].to(self.device)
            label_ids = sample_batched["label_ids"].to(self.args.device)
            label_aspect_ids = aspect_mask

            tag_seq, loss_sa, loss_aspect, loss_at = model(input_ids=input_ids, criteria_index=criteria_index, 
                token_type_ids=segment_ids, labels=label_ids, labels_AR=label_aspect_ids, aspect_mask=aspect_mask)
            loss = loss_sa + loss_aspect * loss_at
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                if args.fp16:
                    # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                    scheduler.step()

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
               
        avg_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0
        return avg_loss, global_step
        
    def train_single_criteria(self, model, optimizer, scheduler, train_data_loader, global_step):
        args = self.args
        tr_loss = 0
        optimizer.zero_grad()
        nb_tr_examples, nb_tr_steps = 0, 0
        criteria_index = 0
        for step, sample_batched in enumerate(tqdm(train_data_loader, desc="Iteration")):
            if args.max_steps > 0 and global_step > args.max_steps:
                break
            reduce_seq_len = seq_reduce_ret_len(sample_batched["input_ids"])
            input_ids = sample_batched["input_ids"][:,:reduce_seq_len].to(self.device)
            aspect_mask = sample_batched["aspect_mask"][:,:reduce_seq_len].to(self.device)
            segment_ids = sample_batched["segment_ids"][:,:reduce_seq_len].to(self.device)
            attention_mask = sample_batched["input_mask"][:,:reduce_seq_len].to(self.device)
            label_ids = sample_batched["label_ids"].to(self.args.device)
            label_aspect_ids = aspect_mask

            tag_seq, loss_sa, loss_aspect, loss_at = model(input_ids=input_ids, criteria_index=criteria_index, 
                token_type_ids=segment_ids, labels=label_ids, labels_AR=label_aspect_ids, aspect_mask=aspect_mask)
            loss = loss_sa 
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
                if args.fp16:
                    # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                    scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
               
        avg_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0
        return avg_loss, global_step

    def train(self, model, optimizer, scheduler, train_data_loader, dev_dataloader, test_dataloader):
        args = self.args
        results = {"bert_model": args.bert_model, "dataset": args.dataset, "warmup":args.warmup_proportion,
                   "batch_size": args.batch_size * args.world_size * args.gradient_accumulation_steps,
                   "learning_rate": args.learning_rate, "seed": args.seed, "num_layers":args.num_layers}
        results["best_checkpoint"] = 0
        results["best_acc_score"] = 0
        results["best_f1_score"] = 0
        results["best_dev_f1_score"] = 0                                
        results["best_checkpoint_path"] = ""

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        num_of_no_improvement = 0
        num_train_optimization_steps = int(
            len(train_data_loader) / self.args.gradient_accumulation_steps) * self.args.num_epoch

        model.train()
        for epoch_num in trange(int(args.num_epoch), desc="Epoch"):
            
            if epoch_num == args.num_epoch_multi_cri:

                logger.info("Loading best checkpoint from stage 1...")
                model = ABSA_CL(tag_size=3, bert_path=args.bert_model, embedding="bert", adv_coefficient=0.06, pooling=args.pooling, encoder=args.encoder, adversary=args.adversary, num_layers=args.num_layers)
                model_path = os.path.join(args.outdir, 'best_checkpoint_stage_1.bin')
                model_dict = torch.load(model_path)
                model.load_state_dict(model_dict)
                
                name_list = ['classifier_at', 'classifier_aspect']
                for name, value in model.named_parameters():
                    if name in name_list:
                        value.requires_grad = False
                params_ = filter(lambda p: p.requires_grad, model.parameters())
                
                if args.fp16:
                    optimizer = FusedAdam(params_,
                                          lr=args.learning_rate,
                                          bias_correction=False)

                    if args.loss_scale == 0:
                        model, optimizer = amp.initialize(model.float(), optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                          loss_scale="dynamic")
                    else:
                        model, optimizer = amp.initialize(model.float(), optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                          loss_scale=args.loss_scale)
                    
                    scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                                      total_steps=num_train_optimization_steps)
                else:
                    optimizer = BertAdam(params_,
                                         lr=args.learning_rate,
                                         warmup=args.warmup_proportion,
                                         t_total=num_train_optimization_steps)
                    
                    scheduler = None
                model = model.to(args.device)
                model.train()
            
            if epoch_num < args.num_epoch_multi_cri:
                _loss, global_step = self.train_multi_criteria(model, optimizer, scheduler, train_data_loader, global_step)
            else:
                _loss, global_step = self.train_single_criteria(model, optimizer, scheduler, train_data_loader, global_step)

            output_dir = os.path.join(args.outdir, "epoch-{}".format(epoch_num))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            checkpoint = output_dir
            result = self._evaluate_acc_f1(model, dev_dataloader)
            model.train()
            if result["f1"] > results["best_f1_score"]:
                results["best_f1_score"] = result["f1"]
                results["best_p_score"] = result["precision"]
                results["best_checkpoint"] = epoch_num
                results["best_checkpoint_path"] = checkpoint
                num_of_no_improvement = 0
                logger.info("Saving models...")
                if epoch_num < args.num_epoch_multi_cri:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_path = os.path.join(args.outdir, 'best_checkpoint_stage_1.bin')
                    torch.save(model_to_save.state_dict(), model_path)
                else:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_path = os.path.join(args.outdir, 'best_checkpoint_stage_2.bin')
                    torch.save(model_to_save.state_dict(), model_path)
            else:
                num_of_no_improvement += 1
            result = {"{}_dev_{}".format(epoch_num, k): v for k, v in result.items()}
            results.update(result)
            output_eval_file = os.path.join(args.outdir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                writer.write(json.dumps(results, ensure_ascii=False))

        loss = _loss
        return loss

def get_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--test_dataset', default=None, type=str)
    parser.add_argument('--data_dir', default='ATB', type=str)
    parser.add_argument('--embedding', default='embedding', type=str)
    parser.add_argument('--encoder', default='transformer', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default='2e-5', type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--bert_dropout', default=0.2, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--log', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=1024, type=int)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--seed', default=50, type=int)
    parser.add_argument('--bert_model', default='./bert-large-uncased', type=str)
    parser.add_argument('--outdir', default='./', type=str)
    parser.add_argument('--tool', default='stanford', type=str)
    parser.add_argument('--warmup_proportion', default=0.06, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--loss_scale', default=0, type=int)
    parser.add_argument('--save', action='store_true', help="Whether to save model")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--rank", type=int, default=0, help="local_rank for distributed training on gpus")
    parser.add_argument("--world_size", type=int, default=1, help="local_rank for distributed training on gpus")
    parser.add_argument("--init_method", type=str, default="", help="init_method")
    parser.add_argument("--pooling", type=str, default="avg_pooling")   
    parser.add_argument('--multi_criteria', action='store_true', help="Using multi criteria")
    parser.add_argument('--adversary', action='store_true', help="Using adversary learning")
    parser.add_argument('--num_epoch_multi_cri', default=20, type=int)
    parser.add_argument("--max_steps", default=-1.0, type=float,
                        help="Total number of training steps to perform.")
    args = parser.parse_args()

    args.initializer = torch.nn.init.xavier_uniform_

    return args

def main():
    args = get_args()

    import datetime
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(args.outdir):
        try:
            os.mkdir(args.outdir)
        except Exception as e:
            print(str(e))
    args.outdir = os.path.join(args.outdir, "{}_bts_{}_lr_{}_warmup_{}_seed_{}_bert_dropout_{}_{}".format(
        args.dataset,
        args.batch_size,
        args.learning_rate,
        args.warmup_proportion,
        args.seed,
        args.bert_dropout,
        now_time
    ))
    if not os.path.exists(args.outdir):
        try:
            os.mkdir(args.outdir)
        except Exception as e:
            print(str(e))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
   
    args.device = device
    args.n_gpu = n_gpu
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}, init_method: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16, args.init_method))

    log_file = '{}/{}-{}.log'.format(args.log, args.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(args)
    ins.run()


if __name__ == '__main__':
    main()
