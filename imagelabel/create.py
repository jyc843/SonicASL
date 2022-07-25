import os
import pdb
from itertools import groupby
import PIL.Image as Image
import cv2
#s = 'HPO:0000234'
import itertools
#[(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)
import os
import shutil

def string2alphabet(name):
    strings = {
        "WHAT"             : "0",
        "WHO"              : "1",
        "WHERE"            : "2",
        "WHEN"             : "3",
        "HOW"              : "4",
        "WHICH"            : "5",
        "SOMEBODY"         : "6",
        "WHY"              : "7",
        "NAME"             : "8",
        "FRIEND"           : "9",
        "TIME"             : "a",
        "UNCLE"            : "b",
        "AUNT"             : "c",
        "CAMERA"           : "d",
        "MIRROR"           : "e",
        "LATE"             : "f",
        "NOTYET"           : "h",
        "HAVE"             : "i",
        "MILK"             : "j",
        "COFFEE"           : "k",
        "WATER"            : "l",
        "SMALL"            : "m",
        "COLD"             : "n",
        "HOT"              : "o",
        "HAPPY"            : "p",
        "SAD"              : "q",
        "CLEAN"            : "r",
        "DIRTY"            : "s",
        "NICE"             : "t",
        "BAD"              : "u",
        "BICYCLE"          : "v",
        "SIGN"             : "w",
        "RUN"              : "x",
        "DISCUSS"          : "y",
        "OPEN"             : "z",
        "YOU"              : "A",
        "QYOU"             : "B",
        "I"                : "C",
        "QI"               : "D",
        "SHE"              : "E",
        "QSHE"             : "F",
        "HERE"             : "G",
        "QHERE"            : "H",
        "NOTHERE"          : "I",
        "NOTBAD"           : "J",
        "THERE"            : "K",
        "QTHERE"           : "L",
        "NOTTHERE"         : "M",
        "SPACE"            : "N",
        "BIG"              : "O",
        "TALL"             : "P",
        "ARROGANT"         : "Q",
        "PLATE"            : "R",
        "CHASPACE"         : "S",
        "CHABIG"           : "T",
        "CHATALL"          : "U",
        "CHAARROGANT"      : "V",
        "CHARROGANT"       : "V",
        "CHAPLATE"         : "W",
        "DRIVE"            : "X",
        "NOTDRIVE"         : "Y",
        "MMDRIVE"          : "Z",
        "THDRIVE"          : "!",
        "CSDRIVE"          : "@",
        "TYPE"             : "#",
        "WRITE"            : "$",
        "NOTTYPE"          : "%",
        "MMTYPE"           : "^",
        "THTYPE"           : "&",
        "CSTYPE"           : "*",
        "READ"             : "(",
        "NOTREAD"          : ")",
        "MMREAD"           : "{",
        "THREAD"           : "[",
        "CSREAD"           : "}",
        "STUDY"            : "]",
        "NOTSTUDY"         : ":",
        "MMSTUDY"          : ";",
        "THSTUDY"          : "<",
        "CSSTUDY"          : ",",
        "CARRY"            : ">",
        "NOTCARRY"         : ".",
        "MMCARRY"          : "?",
        "THCARRY"          : "/",
        "CSCARRY"          : "+"
    }                        
    return strings.get(name)

txt          = 'test.txt';
filePath     = 'subject4/train/';
filePathout  = 'total/';
#os.system('rm -rf total/*');
f            = open(txt,'a');
number       = 0;

names = os.listdir(filePath);

for kk in range(len(names)):
    name1 = filePath + '/' + names[kk];
    (filePathin1, tempfilename1) = os.path.split(name1)
    (filename1, extension)       = os.path.splitext(tempfilename1)
    print('the filename1 and extension are', filename1, extension)
    if extension == '.jpg':
        s1 = filename1
        ss1 = [''.join(list(g)) for k, g in groupby(s1, key=lambda x: x.isdigit())]
        aa  = ss1[2].split('_');
        bb = aa[1:len(aa)-1];
        if len(bb) == 4:
            content1 =  string2alphabet(bb[0]) if string2alphabet(bb[0]) != 'A' else ''
            content2 =  string2alphabet(bb[1]) if string2alphabet(bb[1]) != 'A' else ''
            content3 =  string2alphabet(bb[2]) if string2alphabet(bb[2]) != 'A' else ''
            content4 =  string2alphabet(bb[3]) if string2alphabet(bb[3]) != 'A' else ''
        elif len(bb) == 3:
            content1 =  string2alphabet(bb[0]) if string2alphabet(bb[0]) != 'A' else ''
            content2 =  string2alphabet(bb[1]) if string2alphabet(bb[1]) != 'A' else ''
            content3 =  string2alphabet(bb[2]) if string2alphabet(bb[2]) != 'A' else ''
            content4 =  ''
        elif len(bb) == 2:
            content1 =  string2alphabet(bb[0]) if string2alphabet(bb[0]) != 'A' else ''
            content2 =  string2alphabet(bb[1]) if string2alphabet(bb[1]) != 'A' else ''
            content3 =  ''
            content4 =  ''
        elif len(bb) == 1:
            content1 =  string2alphabet(bb[0]) if string2alphabet(bb[0]) != 'A' else ''
            content2 =  ''
            content3 =  ''
            content4 =  ''
        number = number + 1;
        print(bb,len(bb),content1,content2,content3,content4)

        source       = filePathin1 + '/' + tempfilename1;
        destination  = filePathout + '/' + tempfilename1; # tempfilename;

        shutil.copyfile(source, destination)

        #(filename1, extension) = os.path.splitext(tempfilename1);

        source_csv       = filePathin1 + '/' + filename1 + '_0.csv';
        destination_csv  = filePathout + '/' + filename1 + '_0.csv' ; # tempfilename;
        print('the csv file name is', filename1, source_csv, destination_csv)

        shutil.copyfile(source_csv, destination_csv)

        if len(bb) == 4:
            number = number + 1;
            f.write(tempfilename1)
            f.write(' ')
            f.write(content1)
            f.write(content2)
            f.write(content3)
            f.write(content4)
            f.write('\n')
        elif len(bb) == 3:
            number = number + 1;
            f.write(tempfilename1)
            f.write(' ')
            f.write(content1)
            f.write(content2)
            f.write(content3)
            f.write('\n')
        elif len(bb) == 2:
            number = number + 1;
            f.write(tempfilename1)
            f.write(' ')
            f.write(content1)
            f.write(content2)
            f.write('\n')
        elif len(bb) == 1:
            number = number + 1;
            f.write(tempfilename1)
            f.write(' ')
            f.write(content1)
            f.write('\n')


f.close();
