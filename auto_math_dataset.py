#!/usr/bin/python

'''
@author Mark McMillan
1214401682
CSE576 - Prof. Chitta Baral
python auto_math_dataset.py --input_file MathQA.json --input_type raw --temp QA_temp.json --gen QA_gen.json 
'''

import os
import argparse
import re
import json
from string import Template
import random
from collections import OrderedDict
import spacy
from spacy.matcher import Matcher

en_nlp = spacy.load('en_core_web_sm')

operator_dict = {'divide(': '/', 'add(': '+', 'multiply(': '*', 'subtract(': '-', 'power(': '**', 'speed(': '/'}

def problem_parse(prob_str, ops_list):
    ops = r'\A(divide|add|multiply|subtract|power|speed)\('
    output =''
    open_count = 0
    
    num = re.search(r'\A\d', prob_str)
    if num:
        return prob_str
    
    for i, j in enumerate(prob_str): 
        if j == '(': 
            open_count += 1 
        elif j == ',': 
            if open_count > 1: 
                open_count -= 1
            else:
                q = re.search(ops, prob_str)
                try:
                    split_str = [prob_str[q.end():i], prob_str[i+2:-1]]
                except:
                    # Bad input format
                    output = "N/A"
                    break

                output = "({0} {1} {2})".format(
                    problem_parse(split_str[0], ops_list), 
                    ops_list[q.group()], 
                    problem_parse(split_str[1], ops_list)
                    )
                break
    return output

def create_template(data):
    new_data = []

    for problem in data:
        prob = problem['Problem']
        answer = problem['annotated_formula']

        nums = re.findall(r'\d*\.?\d+', prob)
        nums = list(OrderedDict.fromkeys(nums))

        if len(nums) > 0:
        
            temp = Template('$${n$x}')
            replace_dict = {str(j):(temp.substitute(x = i)) for i, j in enumerate(nums)}

            for k in replace_dict.keys():
                sub = r'\b' + k + r'\b'
                prob = re.sub(sub, replace_dict[k], prob)
                answer = re.sub(sub, replace_dict[k], answer)
            
            # Replace const_
            consts = set(re.findall(r'\bconst_\d+\b', answer))
            for n in consts:
                answer = answer.replace(n, n[6:])

            new_data.append({'Problem': prob, 'Solution': answer})

        # Skip if no numbers
        else:
            break
    return new_data

def gen_problems(data, args):
    low, high, iterations, seed = args.low, args.high, args.data_iter, args.seed
    
    gen_problems = []
    for row in data:
        prob = row['Problem']
        answer_temp = Template(row['Solution'])
        random.seed(seed)

        sub = r'\${n\d+}'
        replace_set = re.findall(sub, prob)
        replace_set = list(OrderedDict.fromkeys(replace_set)) 
        replace = []

        for i in replace_set:
            replace.append(i[2:-1])
        
        prob_temp = Template(prob)

        if len(replace) > 0:
            random_nums = [{x: random.randint(low, high) for x in replace} for _ in range(iterations)]
            for nums in random_nums:
                parsed_sol = problem_parse(answer_temp.substitute(nums), operator_dict)
                try:
                    sol = "{0} = {1}".format(
                        parsed_sol,
                        eval(parsed_sol)
                        )
                except:
                    # Bad input format
                    break

                gen_problems.append({
                    'Problem': prob_temp.safe_substitute(nums),
                    'Solution': sol
                    })
                
        else:
            break

    return gen_problems

def name_replace(data):

    names = ['bob', 'lindsey', 'kathrine', 'kendal', 'katy', 'karen', 'jose', 'john']
    to_replace = []

    matcher = Matcher(en_nlp.vocab)
    pattern = [{'POS': 'PROPN'}]
    matcher.add('NAME', None, pattern)

    for row in data:
        nlp_doc = en_nlp(row['Problem'])
        matches = matcher(nlp_doc)
        matches =set(str(nlp_doc[start:end]) for _, start, end in matches)
        for ent in nlp_doc.ents:
            if (ent.label_ == 'PERSON') and (ent.text in matches):
                    to_replace.append(ent.text)

        temp = row['Problem']
        for i in set(to_replace):
            new_name = names[random.randint(0, len(names)-1)]
            while new_name == i:
                new_name = random.randint(0, len(names)-1)
            temp = re.sub(i, new_name, temp)
        row['Problem'] = temp

    return data
            
# Arg parser

desc = 'This program will take a MathQA dataset json and can produce a\
    templated version of it and/or generate new questions using random values'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument(
    "--input_dir",
    default=None,
    type=str,
    required=False,
    help="Path where to look for input file. Default is current folder"
)
parser.add_argument(
    "--input_file", 
    default=None,
    type=str,
    required=True,
    help="Set input file name"
)
parser.add_argument(
    "--input_type",
    default=None,
    type=str,
    required=True,
    help="Input must be either a ""raw"" dataset or a ""template"" dataset"
)
parser.add_argument(
    "--temp",
    default=None,
    type=str,
    required=False,
    help="Save templated questions as"
)
parser.add_argument(
    "--gen",
    default=None,
    type=str,
    required=False,
    help="Save generated questions as"
)
parser.add_argument(
    "--data_iter",
    default=10,
    type=int,
    required=False,
    help="Set custom amount of generated questions"
)
parser.add_argument(
    "--low",
    default=1,
    type=int,
    required=False,
    help="Lower limit on random numbers used for question generation"
)
parser.add_argument(
    "--high",
    default=300,
    type=int,
    required=False,
    help="Upper limit on random numbers used for question generation"
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    required=False,
    help="Seed used for random numbers used for question generation"
)

args = parser.parse_args()

input_dir = args.input_dir if args.input_dir else "."
input_name = os.path.join(
    input_dir,
    args.input_file
)

if args.input_type not in ['raw', 'template']:
    raise ValueError(
    "Input type must be either ""raw"" or ""template"""
    )

if args.temp:
    temp_name = os.path.join(
                input_dir,
                args.temp
            )

if args.gen:
    gen_name = os.path.join(
                input_dir,
                args.gen
            )

if args.temp or args.gen:

    with open(input_name, 'r') as ifile:
        data = json.load(ifile)
    ifile.close()

    if args.input_type == "raw":
        prob_template = create_template(data)

        if (args.temp and args.gen):
            gen = gen_problems(prob_template, args)

            with open(temp_name, 'w') as tempfile:
                json.dump(prob_template, tempfile, indent=4)
            tempfile.close()

            with open(gen_name, 'w') as genfile:
                json.dump(gen, genfile, indent=4)
            genfile.close()

        elif args.gen:
            gen = gen_problems(prob_template, args)

            with open(gen_name, 'w') as genfile:
                json.dump(gen, genfile, indent=4)
            genfile.close()

        else:
            with open(temp_name, 'w') as tempfile:
                json.dump(prob_template, tempfile, indent=4)
            tempfile.close()

    else:
        gen = gen_problems(data, args)

        with open(gen_name, 'w') as genfile:
                json.dump(gen, genfile, indent=4)
        genfile.close()
else:
    raise ValueError(
    "You must choose to ouput at least one kind of file with --temp or --gen"
    )
