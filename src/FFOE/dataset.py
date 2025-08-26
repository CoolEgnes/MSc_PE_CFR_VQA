"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function

import copy
import os
import json
import _pickle as cPickle
import numpy as np
from torch import nn

import src.utils as utils
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import torch
from torch.utils.data import Dataset
import itertools
COUNTING_ONLY = False

# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering


def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)  # data/gqa/dictionary.pkl
        # idx2word: ['do', 'you', 'see', 'both', ...]; word2idx: {''s': 369, 'a': 11, 'abandoned': 2878, ...}
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        # Dictionary object, store a dictionary word2idx and a list idx2word
        d = cls(word2idx, idx2word)  # <src.FFOE.dataset.Dictionary object at 0x000001F5116C4DA0>
        """
        d = {'idx2word': dict,
             'ntoken': 2931,
             'padding_idx': 2931,
             'word2idx': dict}
        """
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer, entity, teacher_logit):
    if None!=answer:  # {'image_id': 3455, 'labels': [439], 'question_id': '001000', 'scores': [1.0]}
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer,  # {'labels': [439], 'scores': [1.0]}
        'entity'      : entity,  # ['picture', 'in']
        'teacher_logit': teacher_logit}
    return entry

def _load_gqa_dataset(dataroot, args, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'gqa_%s_questions_entities.json' % name)  # gqa_val_questions_entities.json
    # {"questions": [{"image_id": "2405722", "question": "What is this bird called?", "question_id": "05515938", "entities": ["bird"]}, ...}
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    # if 'test' != name[:4]:  # train, val
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)  # 'data/gqa\\cache\\val_target.pkl'
    # [{'image_id': '2405722', 'labels': [1009], 'question_id': '05515938', 'scores': [1.0]}, ...]
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    utils.assert_eq(len(questions), len(answers))  # val: 132062
    entries = []
    # Train and evaluate on tiny sample
    if args.tiny:
        # questions = questions[:30000]
        # answers = answers[:30000]
        questions = questions[:300]
        answers = answers[:300]
    """
    {'answer': {'labels': [439], 'scores': [1.0]}, 
     'entity': ['picture', 'in'], 
     'image': 3882, 
     'image_id': '3455', 
     'question': 'How is the weather in the picture?', 
     'question_id': '001000', 
     'teacher_logit': None}
    """
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entity = question['entities']

        entries.append(_create_entry(img_id2val[img_id], question, answer, entity, None))
    # else:  # test
    #     entries = []
    #     for question in questions:
    #         img_id = question['image_id']
    #         entity = question['entities']
    #         entries.append(_create_entry(img_id2val[img_id], question, None, entity, None))

    return entries

# # use the corresponding code from ViTARC
# # SinusoidalAPE
# class FixedAbsolutePositionalEmbedding(nn.Module):
#     def __init__(self, dim, max_tokens):
#         super().__init__()
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#         t = torch.arange(max_tokens).type_as(inv_freq)
#         # outer product, generates inputs for sinusoidal waves that vary across both positions(t) and frequencies(inv_freq)
#         sinusoid_inp = torch.einsum("i , j -> i j", t, inv_freq)
#         emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
#         self.embed = nn.Embedding.from_pretrained(emb, freeze=True)
#
#     def forward(self, position_ids: torch.Tensor):
#         return self.embed(position_ids.long())

class GQAFeatureDataset(Dataset):
    # def __init__(self, args, name, dictionary, dataroot="/scratch/xcwx3620/MSc_Project/Codes/CFR_VQA-main-zhd/data/gqa/", adaptive=False, train_read=False, train_dset=None):
    # def __init__(self, args, name, dictionary,dataroot="D:/CFR_train_extract_0711/", adaptive=False, train_read=False, train_dset=None):
    def __init__(self, args, name, dictionary, dataroot="./data/", adaptive=False, train_read=False,
                 train_dset=None):
        super(GQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015', 'test']
        # cwd = os.getcwd()
        # print(f"cwd: {cwd}")  # F:\python_code\MSc Project\Baseline\CFR_VQA-main-zhd

        ans2label_path = os.path.join(dataroot, 'cache', 'ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'label2ans.pkl')
        # cPickle: Python object serialization, implements the same algorithm as the Pickle in C instead of python
        # many times faster than python implementation, but not allow subclassing from Pickle
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))  # {'above': 554, 'action figure': 679, 'adidas': 1266, ...}
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))  # ['yes', 'pipe', 'no', ...]
        self.num_ans_candidates = len(self.ans2label)  # 1533
        self.max_boxes = args.max_boxes  # 50
        self.question_len = args.question_len  # 12

        # add arguments about 2D Absolute Position Encoding and OPE
        self.use_ope = args.use_ope
        print(f"self.use_ope: {self.use_ope}")
        self.device = args.device

        self.dictionary = dictionary
        self.adaptive = adaptive
        self.teacher_logits = []
        print('Create %s entries' % name)

        # load stat_word
        # 'val_6_stats_words.json', contains objects in each image
        # self.stat_words = json.load(open("D:/CFR_train_extract_0711/%s_%s_stats_words.json" % (name, args.topk)))  # e.g. "2382986": "cat,brown,towel,bag,box,plastic,container,sandwich,above,green,lettuce,picture,white,cats,color,under,towels,bed,chairs,tan,black,bottom,bags,top,pillow,sandwiches,chair,hair,sticker"
        # self.stat_skip_imgid = json.load(open("D:/CFR_train_extract_0711/%s_%s_stats_skip_imgid.json" % (name, args.topk)))  # contains the id of images that don't have objects
        self.stat_words = json.load(open("./data/%s_%s_stats_words.json" % (name, args.topk)))  # e.g. "2382986": "cat,brown,towel,bag,box,plastic,container,sandwich,above,green,lettuce,picture,white,cats,color,under,towels,bed,chairs,tan,black,bottom,bags,top,pillow,sandwiches,chair,hair,sticker"
        self.stat_skip_imgid = json.load(open("./data/%s_%s_stats_skip_imgid.json" % (name, args.topk)))  # contains the id of images that don't have objects

        self.stat_features = {}

        # load attribute word
        # self.attr_words = json.load(open("D:/CFR_train_extract_0711/%s_attr_words_non_plural_words.json" % name))  # contains the phrases like "black zebra"
        # self.attr_skip_imgid = json.load(open("D:/CFR_train_extract_0711/%s_attr_skip_imgid.json" % name))
        self.attr_words = json.load(open("./data/%s_attr_words_non_plural_words.json" % name))  # contains the phrases like "black zebra"
        self.attr_skip_imgid = json.load(open("./data/%s_attr_skip_imgid.json" % name))

        self.skip_imgid = []
        self.attr_features = {}

        self.ans_list = []
        # val_imgid2idx.pkl
        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))

        # read the image_data.json, to get the width and height of the images, for normalising the bounding boxes
        self.image_data = json.load(open(os.path.join(dataroot, 'image_data.json'), 'rb'))
        # reform the dictionary
        self.new_image_data = {image["image_id"]: {k: v for k, v in image.items() if k != "image_id"} for image in self.image_data}


        # Load image feature
        if name[:4] != 'test':
            h5_path = os.path.join(dataroot, 'ori_train.hdf5')
        else:
            h5_path = os.path.join(dataroot, '%s.hdf5' % name)
        print('loading features from h5 file %s ' % h5_path)
        if not train_read:
            with h5py.File(h5_path, 'r') as hf:
                self.features = np.array(hf.get('image_features'))  # ndarray: (2222393, 2048)
                # Spatial relationships between regions or between regions and the image.
                # Often a combination of image_bb and pos_boxes processed into a fixed-length vector.
                self.spatials = np.array(hf.get('spatial_features'))  # shape: (311827, 6), [x_min, y_min, x_max, y_max, width, height] e.g. [0.         0.00466759 0.6290524  0.62489855 0.6290524  0.620231  ]
                # Positional Box Features, Geometric features derived from bounding boxes (e.g., size, position, aspect ratio)
                # start and end index (inclusive) of the regions in the spatial_features that are relevant to the question
                self.pos_boxes = np.array(hf.get('pos_boxes'))  # shape: (10234, 2), e.g.[ 0 31], means the first 31 bounding boxes are relevant to the question
                # self.bbs = np.array(hf.get('image_bb'))  # shape: (2222393, 4), e.g. # [663.47925 227.9192  735.18164 383.7306 ], [x1,y1,x2,y2], Top-left corner and the bottom-right corner of the bounding box
        else:
            self.features = train_dset.features
            self.spatials = train_dset.spatials
            self.pos_boxes = train_dset.pos_boxes

        # # SinusoidalAPE2D-20250715
        # if args.use_ope:
        #     # If with OPE, half enc is reserved for obj_idx
        #     self.wpe_obj_enc = FixedAbsolutePositionalEmbedding(self.features.shape[1] / 2, max_tokens=100)  # 2048/2 -> 1024
        #     self.wpe_x_enc = FixedAbsolutePositionalEmbedding(self.features.shape[1] / 4, max_tokens=100)  # 2048/4 -> 512
        #     self.wpe_y_enc = FixedAbsolutePositionalEmbedding(self.features.shape[1] / 4, max_tokens=100)  # 2048/4 -> 512
        # else:
        #     self.wpe_x_enc = FixedAbsolutePositionalEmbedding(self.features.shape[1] / 2, max_tokens=100)  # 2048/2 -> 1024
        #     self.wpe_y_enc = FixedAbsolutePositionalEmbedding(self.features.shape[1] / 2, max_tokens=100)  # 2048/2 -> 1024
        #
        # # Add 2D APE
        # self.max_tokens=512
        # self.pos_enc = FixedAbsolutePositionalEmbedding(dim=64, max_tokens=self.max_tokens)
        #
        # # calculate the center of the bounding boxes
        # # x_min = self.spatials[:, 0]
        # # x_max = self.spatials[:, 2]
        # # y_min = self.spatials[:, 1]
        # # y_max = self.spatials[:, 3]
        #
        # self.x_center = (self.spatials[:, 0] + self.spatials[:, 2]) / 2
        # self.y_center = (self.spatials[:, 1] + self.spatials[:, 3]) / 2



        self.entries = _load_gqa_dataset(dataroot, args, name, self.img_id2idx)  # need to check--20250403 17:52, for test set: list: 4237524
        self.tokenize(self.question_len)  # for test_set: 12
        self.stat_word_tokenize_1(args.num_stat_word)  # num_stat_word: 30
        self.attr_word_tokenize(15)
        self.ans_tokenize()  # 'cloudless'
        self.entity_tokenize()  # 'entity': ['picture', 'in']
        self.tensorize()  # convert question_token, entity_token, answer_token, the labels and scores in the value of the key "answer" to tensors
        self.v_dim = self.features.size(1)  # 2048
        self.s_dim = self.spatials.size(1)  # 6

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def entity_tokenize(self, max_length=7):
        """Tokenizes the instruction word.

        This will add entity_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            entity = entry['entity']  # 'entity': ['picture', 'in']
            entity = ' '.join(entity)
            tokens = self.dictionary.tokenize(entity, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding

            entry['entity_token'] = tokens

    def ans_tokenize(self, max_length=2):
        """Tokenizes the answers.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            try:
                ans = self.label2ans[entry['answer']['labels'][0]]
                tokens = self.dictionary.tokenize(ans, False)
            except:
                tokens = []

            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['ans_token'] = tokens

    # Tokenize statistical words 2-gram
    def stat_word_tokenize(self, max_length=40):
        for img_id in self.stat_words:
            words = self.stat_words[img_id]
            # words = words.split(',')
            words = words[:max_length]
            token_words = []
            for word in words:
                tokens = self.dictionary.tokenize(word, False)
                tokens = tokens[:2]
                if len(tokens) < 2:
                    padding = [self.dictionary.padding_idx] * (2 - len(tokens))
                    tokens = tokens + padding
                token_words.append(tokens)
            if len(words) < max_length:
                tmp = list(np.full(2, self.dictionary.padding_idx))
                tmp_token_words = [tmp for _ in range(max_length - len(words))]
                token_words += tmp_token_words
            self.stat_features[img_id] = token_words

    # Tokenize attribute words
    def attr_word_tokenize(self, max_length=15):
        for img_id in self.attr_words:  # {'1001': ['green tree', 'parked car', 'gray building', 'brown dirt', 'blue bench', 'black car', 'tall tree', 'metal cage', 'tall palm trees', 'blue sky', 'palm tree'], ...}
            words = self.attr_words[img_id]
            words = words[:max_length]  # ['gray short', 'yellow frisbee', 'jumping man', 'white logo']
            token_words = []
            for word in words:
                tokens = self.dictionary.tokenize(word, False)
                tokens = tokens[:3]
                if len(tokens) < 3:
                    padding = [self.dictionary.padding_idx] * (3 - len(tokens))
                    tokens = tokens + padding
                token_words.append(tokens)
            if len(words) < max_length:
                tmp = list(np.full(3, self.dictionary.padding_idx))  # Return a new array of given shape and type, filled with fill_value, usually the second parameter.
                tmp_token_words = [tmp for _ in range(max_length - len(words))]
                token_words += tmp_token_words
            self.attr_features[img_id] = token_words

    # Tokenize statistical words
    def stat_word_tokenize_1(self, max_length=40):
        for img_id in self.stat_words:  # self.stat_words: '1001': 'cage,grass,picture,benches,green,front,cars,brown,trees,building,dirt,palm trees,flag,color,round,metal,white,bench,magazines,metallic,sky', ...}
            words = self.stat_words[img_id]
            words = words.split(',')
            words = ' '.join(words)
            tokens = self.dictionary.tokenize(words, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            self.stat_features[img_id] = tokens  # word embedding of the statistical words in the image

    def ans_word_tokenize(self, max_length=2):
        ans_list = []
        for ans in self.label2ans:
            tokens = self.dictionary.tokenize(ans, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            ans_list.append(tokens)
        self.ans_list = ans_list

    def tensorize(self):
        # convert question_token, entity_token, answer_token, the labels and scores in the value of the key "answer" to tensors
        # torch.from_numpy: Creates a Tensor from a numpy.ndarray
        if not torch.is_tensor(self.features):
            self.features = torch.from_numpy(self.features)
            self.spatials = torch.from_numpy(self.spatials)
        for entry in self.entries:
            """
            entry: {'ans_token': tensor([1355, 2931], dtype=torch.int32), 'answer': {'labels': [439], 'scores': [1.0]}, 
            'entity': ['picture', 'in'], 'entity_token': tensor([  21,   14, 2931, 2931, 2931, 2931, 2931], dtype=torch.int32), 'image': 3882, 'image_id': '3455', 
            'q_token': tensor([  95,   22,   15, 1050,   14,   15,   21, 2931, 2931, 2931, 2931, 2931], dtype=torch.int32), 
            'question': 'How is the weather in the picture?', 'question_id': '001000', 'teacher_logit': None}
            """
            question = torch.from_numpy(np.array(entry['q_token']))  # convert the question tokens to tensor, e.g. Tensor: (12,)
            entry['q_token'] = question
            entity = torch.from_numpy(np.array(entry['entity_token']))
            entry['entity_token'] = entity
            ans = torch.from_numpy(np.array(entry['ans_token']))
            entry['ans_token'] = ans

            answer = entry['answer']
            if answer is not None:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None



    def __getitem__(self, index):
        """
        {'ans_token': tensor([1355, 2931], dtype=torch.int32), 'answer': {'labels': tensor([439], dtype=torch.int32),
         'scores': tensor([1.])}, 'entity': ['picture', 'in'], 'entity_token': tensor([  21,   14, 2931, 2931, 2931, 2931, 2931],
         dtype=torch.int32), 'image': 3882, 'image_id': '3455',
         'q_token': tensor([  95,   22,   15, 1050,   14,   15,   21, 2931, 2931, 2931, 2931, 2931], dtype=torch.int32),
         'question': 'How is the weather in the picture?', 'question_id': '001000', 'teacher_logit': None}
        """
        entry = self.entries[index]
        features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        features = features[:self.max_boxes]
        spatials = spatials[:self.max_boxes]

        question = entry['q_token']  # tensor
        sent = entry['question']  # 'How is the weather in the picture?'
        entity = entry['entity_token']  # tensor
        question_id = entry['question_id']  # '001000'
        answer = entry['answer']  # {'labels': tensor([439], dtype=torch.int32), 'scores': tensor([1.])}
        img_id = str(entry['image_id'])  # '3455'
        stat_features = torch.from_numpy(np.array(self.stat_features[img_id]))  # tensor
        attr_features = torch.from_numpy(np.array(self.attr_features[img_id]))  # tensor
        ans = entry['ans_token']  # 'ans_token': tensor([1355, 2931], dtype=torch.int32)

        # print(f"features.shape: {features.shape}")
        # print(f"spatials.shape: {spatials.shape}")
        # print(f"img_id: {img_id}")
        # print(f"stat_features.shape: {stat_features.shape}")
        # print(f"entity.shape: {entity.shape}")
        # print(f"attr_features.shape: {attr_features.shape}")
        # print(f"question.shape: {question.shape}")
        # print(f"len(sent): {len(sent)}")
        # print(f"target.shape: {target.shape}")
        # print(f"ope.shape: {ope.shape}")
        # print(f"ans.shape: {ans.shape}")

        # # Add 2D APE
        # # Discretize coordinates
        # position_ids = (spatials[:, :4] * (self.max_tokens - 1)).long()  # [N,4]
        # # Encode the coordinates
        # ape_2d = torch.cat([
        #     self.pos_enc(position_ids[:, 0]),  # x_min
        #     self.pos_enc(position_ids[:, 1]),  # y_min
        #     self.pos_enc(position_ids[:, 2]),  # x_max
        #     self.pos_enc(position_ids[:, 3])  # y_max
        # ], dim=-1)  # Tensor: (31, 256), for test: Tensor: (40, 256)
        # # concatenate the spatial_feature and 2D APE
        # spatials = torch.cat([spatials,  # [31,6], for test: [40,6]
        #                       ape_2d],          # [31,256]
        #                     dim=-1)
        # print(f"spatials.shape: {spatials.shape}")  # Tensor: [31,262]

        # Add OPE--20250722
        # if self.use_ope:
        #     object_indices = np.arange(0, self.pos_boxes[entry['image']][1] - self.pos_boxes[entry['image']][0])  # should be np.arange(0,36)
        #     if len(object_indices) > self.max_boxes:
        #         object_indices = np.arange(0, self.max_boxes)
        #     # ope = self.get_sinusoid_ope(object_indices, 768 // 2)  # (box_num, 384), compatible with the output size of the visn_fc, Linear(in=2048, out=768)
        #     # ope = ope.to(torch.float32)
        #     indices = torch.arange(len(object_indices)).unsqueeze(1)  # Shape: (50, 1)
        #     spatials = torch.cat([indices, spatials], dim=1)
        

        if answer is not None:
            labels = answer['labels']  # tensor([439], dtype=torch.int32)
            scores = answer['scores']  # tensor([1.])
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                # print(f"type(labels): {type(labels)}")  # <class 'torch.Tensor'>
                # print(f"type(scores): {type(scores)}")  # <class 'torch.Tensor'>
                # target: Tensor: (1533,)
                """
                Tensor.scatter_(dim, index, src, *, reduce=None) → Tensor
                Writes all values from the tensor src into self at the indices specified in the index tensor. 
                For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
                For a 3-D tensor, self is updated as:
                self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
                self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
                self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
                """
                # assemble the target tensor, e.g. labels=Tensor([439]), target[439]=1.0
                target.scatter_(0, labels.long(), scores)  # value=scores.item()
            # 20250703: also return the imgid
            # return features, spatials, stat_features, entity, attr_features, question, sent, target, img_id, ope, ans
            return features, spatials, stat_features, entity, attr_features, question, sent, target, img_id, ans
        else:
            # return features, spatials, stat_features, entity, attr_features, question, sent, question_id, img_id, ope, ans
            return features, spatials, stat_features, entity, attr_features, question, sent, question_id, img_id, ans

    def __len__(self):
        return len(self.entries)
