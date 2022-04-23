#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
def main():
    """
    mapper
    """
    for line in sys.stdin:
        data = line.strip().split(",")
        climate_zone = data[3]
        if(climate_zone != "climate_zone"): #not count the header of csv
            print("\t".join([climate_zone, "1"]))
#if __name__ == "__main__":
#    main()
main()