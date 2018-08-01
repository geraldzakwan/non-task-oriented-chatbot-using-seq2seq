from nltk.tokenize import word_tokenize

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
    word_count_pertanyaan = 0
    word_count_jawaban = 0
    max_word_jawaban = 0
    max_word_pertanyaan = 0
    kalimat_pertanyaan_max = ""
    kalimat_jawaban_max = ""

    i = 0

    with open(path, 'r', encoding="latin1") as in_file:
        for line in in_file:
            if(line != "===\n"):
                line = line.strip('\n')
                tokens = word_tokenize(line)
                # print(str(i) + ' ' + line)

                if(i % 2 == 0):
                    word_count_pertanyaan = word_count_pertanyaan + len(tokens)
                    if(len(tokens) > max_word_pertanyaan):
                        max_word_pertanyaan = len(tokens)
                        kalimat_pertanyaan_max = line
                else:
                    word_count_jawaban = word_count_jawaban + len(tokens)
                    if(len(tokens) > max_word_jawaban):
                        max_word_jawaban = len(tokens)
                        kalimat_jawaban_max = line

                i = i + 1

    lines_count_arr.append((path, word_count_pertanyaan, word_count_jawaban, max_word_pertanyaan, max_word_jawaban, i, kalimat_pertanyaan_max, kalimat_jawaban_max))

print('Output file : ')
path = input()

with open(path, 'w') as out_file:
    for info in lines_count_arr:
        for inf in info:
            out_file.write(str(inf) + ' ')
        out_file.write('\n')
