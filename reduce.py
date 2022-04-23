#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import groupby
from operator import itemgetter
import sys


def read_mapper_output(std_input, separator='\t'):
    for line in std_input:
        yield line.rstrip().split(separator, 1)


def main(separator='\t'):
    data = read_mapper_output(sys.stdin, separator=separator)
    for current_word, group in groupby(data, itemgetter(0)):
        try:
            total_count = sum(int(count) for current_word, count in group)
            print('{}{}{}'.format(current_word, separator, total_count))
        except ValueError:
            pass

if __name__ == '__main__':
    main()