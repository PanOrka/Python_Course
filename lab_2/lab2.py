import os
import hashlib
import sys


def text_count(file):
    try:
        print("Amount of bytes:", (os.stat(file)).st_size)
    except(FileNotFoundError):
        print("File doesn't exist")
        exit(1)

    with open(file) as f:
        words = 0
        lines = 0
        max_line = 0
        for line in f:
            if len(line) > max_line:
                max_line = len(line) - 1 # \n
            if line.endswith(".\n"):
                words += line.count(" ") + 1
            else:
                words += line.count(" ")
            lines += 1
        print("Words:", words)
        print("Lines:", lines)
        print("Max lenght of line:", max_line)


def base_64_encode(from_file, to_file):
    table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    with open(from_file) as f:
        with open(to_file, mode='w') as t:
            three_bytes = f.read(3) # ASCII wiec czytamy po 1 byte
            while three_bytes:
                chunk = "".join("{0:{fill}{align}8}".format(format(ord(x), 'b'), fill='0', align='>') for x in three_bytes) # ord() -> binary i align do 8 bitow
                start = 0
                end = 6
                while end <= len(chunk):
                    t.write(table[int(chunk[start:end:1], 2)])
                    start += 6
                    end += 6
                end = end - len(chunk)
                if end != 6:
                    t.write(table[int(chunk[start::1] + end*"0", 2)])
                    t.write("="*(end//2))
                three_bytes = f.read(3)


def base_64_decode(from_file, to_file):
    table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    with open(from_file) as f:
        with open(to_file, mode='w') as t:
            four_sixbits = f.read(4) # czytamy po 4 bo 4*6 = 24 => 24%8 = 0
            while four_sixbits:
                chunk = "".join("{0:{fill}{align}6}".format(bin(table.index(x)).replace("0b", ""), fill='0', align='>') for x in four_sixbits.replace("=", ""))
                start = 0
                end = 8
                while end <= len(chunk):
                    t.write(chr(int(chunk[start:end:1], 2)))
                    start += 8
                    end += 8
                four_sixbits = f.read(4)


def change_to_lower(path = "./test"):
    for root, dirs, files in os.walk(path, topdown=False):
        for d in dirs:
            os.rename(root + "/" + d, root + "/" + d.lower())
        for f in files:
            os.rename(root + "/" + f, root + "/" + f.lower())


class file:
    was_taken = False
    def __init__(self, path):
        self.path = path


def hash_func(path1, path2):
    size = 60000
    with open(path1, "rb") as fa1:
        with open(path2, "rb") as fa2:
            content1 = fa1.read(size)
            while (content1):
                if hashlib.md5(content1).hexdigest() == hashlib.md5(fa2.read(size)).hexdigest():
                    content1 = fa1.read(size)
                else:
                    return False
    return True
            


def repchecker(path = "./test"):
    all_files = []
    same_files = []
    for root, _, files in os.walk(path):
        for f in files:
            all_files += [file(root + "/" + f)]
    for f in all_files:
        if not f.was_taken:
            f.was_taken = True
            temp_same = [f]
            for f2 in all_files:
                if not f2.was_taken:
                    try:
                        if os.stat(f.path).st_size == os.stat(f2.path).st_size: # size + funkcja hash
                            if hash_func(f.path, f2.path):
                                temp_same += [f2]
                                f2.was_taken = True
                    except FileNotFoundError:
                        print(sys.exc_info()[0])
            if len(temp_same) > 1:
                same_files.append(temp_same)
    for f in same_files:
        print("----------------------------------")
        for k in f:
            print(k.path)
