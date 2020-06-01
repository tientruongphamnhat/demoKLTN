from nltk.translate.bleu_score import sentence_bleu
import argparse


def argparser():
    Argparser = argparse.ArgumentParser()
    Argparser.add_argument('--reference', type=str,
                           default='summaries.txt', help='Reference File')
    Argparser.add_argument('--candidate', type=str,
                           default='candidates.txt', help='Candidate file')

    args = Argparser.parse_args()
    return args


args = argparser()

reference = open(args.reference, 'r', encoding='UTF-8').readlines()
candidate = open(args.candidate, 'r', encoding='UTF-8').readlines()

if len(reference) != len(candidate):
    raise ValueError('The number of sentences in both files do not match.')

score = 0.

for i in range(len(reference)):
    score += sentence_bleu([reference[i].strip().split()],
                           candidate[i].strip().split())

score /= len(reference)
print("The bleu score is: "+str(score))


# python file_name.py --reference file1.txt --candidate file2.txt
