import os

# model_list = []

# model_list.append(['uniCharLSTM_50', 'uniCharLSTM_100', 'uniCharLSTM_150', 'uniCharLSTM_200'])
# model_list.append(['biCharLSTM_50', 'biCharLSTM_100', 'biCharLSTM_150', 'biCharLSTM_200'])
# model_list.append(['uniLSTM_25', 'uniLSTM_50', 'uniLSTM_75', 'uniLSTM_100', 'uniLSTM_125', 'uniLSTM_150', 'uniLSTM_175', 'uniLSTM_200'])
# model_list.append(['biLSTM_25', 'biLSTM_50', 'biLSTM_75', 'biLSTM_100', 'biLSTM_125', 'biLSTM_150', 'biLSTM_175', 'biLSTM_200'])

# model_list.append('rnn')
# model_list.append('lstm')
# model_list.append('gru')
# model_list.append('birnn')
# model_list.append('bilstm')
# model_list.append('bigru')
# model_list.append('attrnn')
# model_list.append('attlstm')
# model_list.append('attgru')
# model_list.append('attbirnn')
# model_list.append('attbilstm')
# model_list.append('attbigru')
# model_list.append('charrnn')
# model_list.append('charlstm')
# model_list.append('chargru')
# model_list.append('charbirnn')
# model_list.append('charbilstm')
# model_list.append('charbigru')

# for model in model_list:
#     os.system("python chatbot_web/app_demos.py " + model + ' bleu')

for i in range(25, 26):
    os.system("python chatbot_web/app_demos.py " + str(i) + ' bleu')
