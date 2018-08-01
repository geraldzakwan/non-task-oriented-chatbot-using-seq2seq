import os
import sys

# arg-1 : source file
# arg-2 : target file
# arg-3 : number of lines to be copied

lines_array = []

with open(sys.argv[1], 'r', encoding="latin1") as in_file:
    i = 0
    for line in in_file:
        if(i < int(sys.argv[3])):
            lines_array.append(line)
        else:
            break
        i = i + 1

with open(sys.argv[2], 'w') as out_file:
    i = 0
    err = 0
    while (i < len(lines_array)):
        try:
            out_file.write(lines_array[i])
        except:
            err = err + 1
            i = i + 1

            print("Error on line:" + str(i))
            print("Total error:" + str(err))
        i = i + 1
