import torch
charDic={' ':0,
         "'":1,
         '+':2,
         '-':3,
         '.':4,
         'A':5,
         'B':6,
         'C':7,
         'D':8,
         'E':9,
         'F':10,
         'G':11,
         'H':12,
         'I':13,
         'J':14,
         'K':15,
         'L':16,
         'M':17,
         'N':18,
         'O':19,
         'P':20,
         'Q':21,
         'R':22,
         'S':23,
         'T':24,
         'U':25,
         'V':26,
         'W':27,
         'X':28,
         'Y':29,
         'Z':30,
         '_':31,
         '@':32,
         '?':33}
char_List=[' ',"'",'+','-','.','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','_','@','?']
print(len(char_List))

def to_string(list):
    ss=""
    for i in range(list.size(1)):
        index=torch.argmax(list[0][i])
        ss=ss+char_List[index]
    return ss