stop = False
path_list = []

while(not stop):
    print("Filename : ")
    path = input()

    if(path != 'q'):
        path_list.append(path)
    else:
        stop = True

total_char_pertanyaan = 0
total_char_jawaban = 0
total_pertanyaan = 0
total_jawaban = 0
max_char_pertanyaan = 0
max_char_jawaban = 0

for path in path_list:
    with open(path, 'r', encoding="latin1") as in_file:
        for line in in_file:
            splitted = line.split(' ')

            if('cornell_cleaned.txt' not in splitted[0] and 'reddit' not in splitted[0]):
                total_char_pertanyaan = total_char_pertanyaan + int(splitted[1])
                total_char_jawaban = total_char_jawaban + int(splitted[2])
                total_pertanyaan = total_pertanyaan + int(splitted[5])/2
                total_jawaban = total_jawaban + int(splitted[5])/2

                if(int(splitted[3]) > max_char_pertanyaan):
                    max_char_pertanyaan = int(splitted[3])
                if(int(splitted[4]) > max_char_jawaban):
                    max_char_jawaban = int(splitted[4])

print('Rata2 char pertanyaan : ' + str((total_char_pertanyaan/total_pertanyaan)))
print('Rata2 char jawaban : ' + str((total_char_jawaban/total_jawaban)))
print('Rata2 char kalimat : ' + str(((total_char_pertanyaan + total_char_jawaban)/(total_pertanyaan + total_jawaban))))
print('Max char pertanyaan : ' + str(max_char_pertanyaan))
print('Max char jawaban : ' + str(max_char_jawaban))
