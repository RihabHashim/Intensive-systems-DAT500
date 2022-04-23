#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import json
def main():
    """
    mapper
    """
    for line in sys.stdin:
        inputJson = json.loads(line)
        climate_zone = inputJson.get("climate_zone")
        print("\t".join([climate_zone, "1"]))
#if __name__ == "__main__":
#    main()
main()