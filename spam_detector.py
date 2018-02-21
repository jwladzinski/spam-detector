# -*- coding: utf-8 -*-

import sys
import json


def main():
    config = json.loads(open(sys.argv[1]).read())
    print(config)


if __name__ == '__main__':
    main()