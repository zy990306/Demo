# # 评测文本生成的质量
#
#
# from nltk.translate.bleu_score import corpus_bleu
# from nltk.translate.meteor_score import single_meteor_score
#
# from pycocoevalcap.cider import cider_scorer
# from pycocoevalcap.spice import spice
# from pycocoevalcap.rouge import rouge
# from pycocoevalcap.bleu import bleu_scorer
# from pycocoevalcap.meteor import meteor
# from rouge import Rouge
#
# from pycocoevalcap.cider.cider import Cider
# # from pycocoevalcap.spice import Spice
# # from pycocoevalcap.rouge import Rouge
# # from pycocoevalcap.bleu import Bleu
# # from pycocoevalcap.meteor import Meteor
#
# # CIDEr指标评估函数
# def evaluate_cider(ref, cand):
#     cider_scorer = Cider()
#     cider_score, _ = cider_scorer.compute_score(ref, cand)
#     return cider_score
#
# # SPICE指标评估函数
# # def evaluate_spice(ref, cand):
# #     spice_scorer = Spice()
# #     spice_score, _ = spice_scorer.compute_score(ref, cand)
# #     return spice_score
# #
# # # ROUGE指标评估函数
# # def evaluate_rouge(ref, cand):
# #     rouge_scorer = Rouge()
# #     rouge_score, _ = rouge_scorer.compute_score(ref, cand)
# #     return rouge_score
# #
# # # BLEU指标评估函数
# # def evaluate_bleu(ref, cand):
# #     bleu_scorer = Bleu()
# #     bleu_score, _ = bleu_scorer.compute_score(ref, cand)
# #     return bleu_score
# #
# # # METEOR指标评估函数
# # def evaluate_meteor(ref, cand):
# #     meteor_scorer = Meteor()
# #     meteor_score, _ = meteor_scorer.compute_score(ref, cand)
# #     return meteor_score
#
#
#
# # def calculate_bleu(reference, candidate):
# #     reference = [reference.split()]
# #     candidate = candidate.split()
# #     return bleu_scorer.BleuScorer.compute_score(reference, candidate)
# #
# # def calculate_meteor(reference, candidate):
# #     return meteor.Meteor._score(reference, candidate)
# #
# # def calculate_rouge(reference, candidate):
# #     rouge = Rouge()
# #     rouge_score = rouge.get_scores(reference, candidate, avg=True)  # a和b里面包含多个句子的时候用
# #     rouge_score1 = rouge.get_scores(reference, candidate)  # a和b里面只包含一个句子的时候用
# #     # 以上两句可根据自己的需求来进行选择
# #     r1 = rouge_score["rouge-1"]
# #     r2 = rouge_score["rouge-2"]
# #     rl = rouge_score["rouge-l"]
# #
# #     return r1, r2, rl
# #
# # def calculate_cider(reference, candidate):
# #     cider_scorer = Cider()
# #     cider_score, _ = cider_scorer.compute_score({'caption': [reference]}, {'caption': [candidate]})
# #     return cider_score
# #
# # def calculate_spice(reference, candidate):
# #     spice_scorer = Spice()
# #     spice_score, _ = spice_scorer.compute_score({'caption': [reference]}, {'caption': [candidate]})
# #     return spice_score
#
# def main():
#     a = ["i am a student from china"]  # 预测摘要
#     b = ["i am student from school on japan"]  # 参考摘要
#     # a = ["i am a student from china", "the cat was found under the bed"]  # 预测摘要
#     # b = ["i am student from school on japan", "the cat was under the bed"]  # 参考摘要
#     # bleu = calculate_bleu(a,b)
#     # meteor = calculate_meteor(a,b)
#     # r1, r2, rl = calculate_rouge(a,b)
#     cider = evaluate_cider(a,b)
#     print(cider)
#     # spice = calculate_spice(a,b)
#     # print(r1)
#     # print(r2)
#     # print(rl)
#     # print(f'blue:{bleu},meteor:{meteor},rouge1:{r1},rouge2:{r2},rougel:{rl},cider:{cider},spice:{spice}')
# if __name__ == '__main__':
#     main()
# -*- coding=utf-8 -*-
# author: w61
# Test for several ways to compute the score of the generated words.
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))


if __name__ == '__main__':
    # ref = {
    #     '1': ['go down the stairs and stop at the bottom .'],
    #     '2': ['this is a cat.']
    # }
    # gt = {
    #     '1': ['Walk down the steps and stop at the bottom. '],
    #     '2': ['It is a cat.']
    # }
    ref = {
        '1': ['go down the stairs and stop at the bottom .']
    }
    gt = {
        '1': ['Walk down the steps and stop at the bottom. ']
    }
    # 注意，这里如果只有一个sample，cider算出来会是0。
    scorer = Scorer(ref, gt)
    scorer.compute_scores()
