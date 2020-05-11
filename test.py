#!/usr/bin/python

import sys
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

def gen_problems(data, low=1, high=300, iterations=1, seed=42):

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

desc = 'This program will take a MathQA dataset json and can produce a\
    templated version of it and/or generate new questions using random values'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument("--input", "-i", help="set input file name")
parser.add_argument("--temp", "-t", help="save template file as", action='store_true')
parser.add_argument("--gen", "-g", help="save generated new questions as", action='store_true')

args = parser.parse_args()

name = 'dev'

with open(name + '.json', 'r') as ifile:
    data = json.load(ifile)
ifile.close()

wut = name_replace(data)

prob_template = create_template(wut)


gen = gen_problems(prob_template)

with open(name + '_template.json', 'w') as outfile:
    json.dump(prob_template, outfile, indent=4)
outfile.close()


with open(name + '_gen.json', 'w') as outfile:
    json.dump(gen, outfile, indent=4)
outfile.close()
