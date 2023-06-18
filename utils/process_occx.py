import sys
import os, os.path as osp
import json

sys.path.insert(1, os.getcwd())

from parsers import DocxParser

infile = '/opt/collections/oneclickcx/How to connect external display.docx'
# infile = '/home/sergey/test_doc.docx'

dp = DocxParser(1024)
content, meta = dp.process_file(infile)
print(content)

# with open('temp_doc2json.json','wt') as f:
#     json.dump(struct, f, indent=4)