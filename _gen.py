
import os
d=os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(d,'server.py'),'w',encoding='utf-8') as f:
    f.write(open(os.path.join(d,'_src.txt'),'r',encoding='utf-8').read())
print('done')
