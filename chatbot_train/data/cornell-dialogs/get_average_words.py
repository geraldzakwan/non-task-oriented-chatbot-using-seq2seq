stop = False
path_list = []

while(not stop):
    print("Filename : ")
    path = input()

    if(path != 'q'):
        path_list.append(path)
    else:
        stop = True

total_word_pertanyaan = 0
total_word_jawaban = 0
total_pertanyaan = 0
total_jawaban = 0
max_word_pertanyaan = 0
max_word_jawaban = 0

for path in path_list:
    with open(path, 'r', encoding="latin1") as in_file:
        for line in in_file:
            splitted = line.split(' ')

            if('cornell_cleaned.txt' not in splitted[0] and 'reddit' not in splitted[0]):
                total_word_pertanyaan = total_word_pertanyaan + int(splitted[1])
                total_word_jawaban = total_word_jawaban + int(splitted[2])
                total_pertanyaan = total_pertanyaan + int(splitted[5])/2
                total_jawaban = total_jawaban + int(splitted[5])/2

                if(int(splitted[3]) > max_word_pertanyaan):
                    max_word_pertanyaan = int(splitted[3])
                if(int(splitted[4]) > max_word_jawaban):
                    max_word_jawaban = int(splitted[4])

print('Rata2 word pertanyaan : ' + str((total_word_pertanyaan/total_pertanyaan)))
print('Rata2 word jawaban : ' + str((total_word_jawaban/total_jawaban)))
print('Rata2 word kalimat : ' + str(((total_word_pertanyaan + total_word_jawaban)/(total_pertanyaan + total_jawaban))))
print('Max word pertanyaan : ' + str(max_word_pertanyaan))
print('Max word jawaban : ' + str(max_word_jawaban))
