stop = False
path_list = []

while(not stop):
    print("Filename : ")
    path = input()

    if(path != 'q'):
        path_list.append(path)
    else:
        stop = True

lines_count_arr = []

for path in path_list:
    char_count_pertanyaan = 0
    char_count_jawaban = 0
    max_char_jawaban = 0
    max_char_pertanyaan = 0
    kalimat_pertanyaan_max = ""
    kalimat_jawaban_max = ""

    i = 0

    with open(path, 'r', encoding="latin1") as in_file:
        for line in in_file:
            if(line != "===\n"):
                line = line.strip('\n')
                # print(str(i) + ' ' + line)

                if(i % 2 == 0):
                    char_count_pertanyaan = char_count_pertanyaan + len(line)
                    if(len(line) > max_char_pertanyaan):
                        max_char_pertanyaan = len(line)
                        kalimat_pertanyaan_max = line
                else:
                    char_count_jawaban = char_count_jawaban + len(line)
                    if(len(line) > max_char_jawaban):
                        max_char_jawaban = len(line)
                        kalimat_jawaban_max = line

                i = i + 1

    lines_count_arr.append((path, char_count_pertanyaan, char_count_jawaban, max_char_pertanyaan, max_char_jawaban, i, kalimat_pertanyaan_max, kalimat_jawaban_max))

print('Output file : ')
path = input()

with open(path, 'w') as out_file:
    for info in lines_count_arr:
        for inf in info:
            out_file.write(str(inf) + ' ')
        out_file.write('\n')
