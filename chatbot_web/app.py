from cornell_word_seq2seq_glove_predict import CornellWordGloveChatBot
from cornell_word_seq2seq_glove_predict_gru import CornellWordGloveChatBotGRU
from cornell_char_seq2seq_predict import CornellCharChatBot
from gunthercox_word_seq2seq_glove_predict import GunthercoxWordGloveChatBot
from twitter_word_seq2seq_glove_predict import TwitterWordGloveChatBot

import os
import sys
import nltk

from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort, session
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, PostbackEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage, ButtonsTemplate,
    URITemplateAction, TemplateSendMessage
)

from urllib import parse, request as req
import jwt, json

load_dotenv(find_dotenv(), override=True)
app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

type = sys.argv[1]
dataset = sys.argv[2]
topology = sys.argv[3]

if(type == 'char'):
    cornell_char_chat_bot = CornellCharChatBot()
elif(dataset == 'cornell'):
    if(topology == 'gru'):
        cornell_word_glove_chat_bot = CornellWordGloveChatBotGRU(type)
    else:
        cornell_word_glove_chat_bot = CornellWordGloveChatBot(type)
elif(dataset == 'gunthercox'):
    gunthercox_word_glove_chat_bot = GunthercoxWordGloveChatBot(type)
elif(dataset == 'twitter'):
    twitter_word_glove_chat_bot = TwitterWordGloveChatBot(type)

print('Load model succesful')

cornell_word_glove_chat_bot_conversations = []
gunthercox_word_glove_chat_bot_conversations = []
twitter_word_glove_chat_bot_conversations = []

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)

if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

chatbot_state = {}
chatbot_state['dataset'] = 'cornell'

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def message_text(event):
    in_text = event.message.text.lower()
    user_id = event.source.user_id

    if('set dataset to' in in_text):
        tokens = nltk.tokenize.word_tokenize(in_text)
        dataset_name = tokens[len(tokens) - 1]
        chatbot_state['dataset'] = dataset_name
        res_text = 'You are now talking with ' + dataset_name + ' chatbot'
    else:
        response_data = reply(in_text, 'word-glove', chatbot_state['dataset'])
        print(response_data['reply'])
        res_text = response_data['reply']

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=res_text)
    )

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return 'About Us'

@app.route('/gunthercox_word_glove_reply', methods=['POST', 'GET'])
def gunthercox_word_glove_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            gunthercox_word_glove_chat_bot_conversations.append('YOU: ' + sent)
            reply = gunthercox_word_glove_chat_bot.reply(sent)
            gunthercox_word_glove_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('gunthercox_word_glove_reply.html', conversations=gunthercox_word_glove_chat_bot_conversations)

@app.route('/cornell_word_glove_reply', methods=['POST', 'GET'])
def cornell_word_glove_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            cornell_word_glove_chat_bot_conversations.append('YOU: ' + sent)
            reply = cornell_word_glove_chat_bot.reply(sent)
            cornell_word_glove_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('cornell_word_glove_reply.html', conversations=cornell_word_glove_chat_bot_conversations)

@app.route('/twitter_word_glove_reply', methods=['POST', 'GET'])
def twitter_word_glove_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            twitter_word_glove_chat_bot_conversations.append('YOU: ' + sent)
            reply = twitter_word_glove_chat_bot.reply(sent)
            twitter_word_glove_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('twitter_word_glove_reply.html', conversations=twitter_word_glove_chat_bot_conversations)

@app.route('/chatbot_reply', methods=['POST', 'GET'])
def chatbot_reply():
    if request.method == 'POST':
        if not request.json or 'sentence' not in request.json or 'level' not in request.json or 'dialogs' not in request.json:
            abort(400)
        sentence = request.json['sentence']
        level = request.json['level']
        dialogs = request.json['dialogs']
    else:
        sentence = request.args.get('sentence')
        level = request.args.get('level')
        dialogs = request.args.get('dialogs')

    target_text = sentence
    if level == 'word-glove' and dialogs == 'cornell':
        target_text = cornell_word_glove_chat_bot.reply(sentence)
    elif level == 'word-glove' and dialogs == 'gunthercox':
        target_text = gunthercox_word_glove_chat_bot.reply(sentence)
    return jsonify({
        'sentence': sentence,
        'reply': target_text,
        'dialogs': dialogs,
        'level': level
    })

def reply(sentence, level, dialogs):
    target_text = sentence

    if level == 'word-glove' and dialogs == 'cornell':
        target_text = cornell_word_glove_chat_bot.reply(sentence)

    elif level == 'word-glove' and dialogs == 'gunthercox':
        target_text = gunthercox_word_glove_chat_bot.reply(sentence)

    elif level == 'word-glove' and dialogs == 'twitter':
        target_text = twitter_word_glove_chat_bot.reply(sentence)

    elif level == 'char' and dialogs == 'cornell':
        target_text = cornell_char_chat_bot.reply(sentence)

    return {
        'sentence': sentence,
        'reply': target_text,
        'dialogs': dialogs,
        'level': level
    }

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

def get_inputs_and_references(dataset_folder_name, sample_amount):
    path = "chatbot_train/data/" + dataset_folder_name + "/test.txt"

    with open(path) as f:
        lines = f.readlines()

        i = 1
        inputs = []
        references = []

        for line in lines:
            if (i % 2 == 1):
                inputs.append(line[0:len(line)-1])
            else:
                references.append(line)

            if(i == sample_amount * 2):
                break

            i = i + 1

        return {
            'inputs' : inputs,
            'references' : references
        }

def get_outputs_and_references(dataset_folder_name, sample_amount):
    data = get_inputs_and_references(dataset_folder_name, sample_amount)

    inputs = data['inputs']
    references = data['references']
    outputs = []

    if(dataset_folder_name == 'cornell-dialogs'):
        dataset = 'cornell'
    else:
        dataset = dataset_folder_name

    data = []
    for i in range(0, len(inputs)):
        try:
            if(type == 'char'):
                data.append({
                    'output' : reply(inputs[i], 'char', dataset)['reply'],
                    'reference' : references[i]
                })
            else:
                data.append({
                    'output' : reply(inputs[i], 'word-glove', dataset)['reply'],
                    'reference' : references[i]
                })
        except:
            print("One data skipped because of shape error")

    return data

# CALCULATE BLEU SCORE
def bleu_score(dialogs, sample_amount):
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.bleu_score import SmoothingFunction
    smoothie = SmoothingFunction().method4

    if(dialogs == 'cornell'):
        dataset_folder_name = 'cornell-dialogs'
    else:
        dataset_folder_name = dialogs

    data = get_outputs_and_references(dataset_folder_name, sample_amount)
    # print('DATAAA')
    # print(data)

    sum_bleu_score_1 = 0
    sum_bleu_score_2 = 0
    sum_bleu_score_3 = 0
    sum_bleu_score_4 = 0

    i = 0

    for datum in data:
        reference = []
        reference_str = datum['reference'].lower()
        output_str = datum['output']

        reference_str = reference_str.replace(".", "")
        reference_str = reference_str.replace(",", "")
        output_str = output_str.replace(".", "")
        output_str = output_str.replace(",", "")

        reference.append(nltk.tokenize.word_tokenize(reference_str))
        candidate = nltk.tokenize.word_tokenize(output_str)
        print(reference)
        print(candidate)

        try:
            bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        except:
            bleu_1 = 0.5

        try:
            bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        except:
            bleu_2 = 0.5

        try:
            bleu_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        except:
            bleu_3 = 0.5

        try:
            bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        except:
            bleu_4 = 0.5

        print(bleu_1)
        print(bleu_2)
        print(bleu_3)
        print(bleu_4)

        sum_bleu_score_1 = sum_bleu_score_1 + bleu_1
        sum_bleu_score_2 = sum_bleu_score_2 + bleu_2
        sum_bleu_score_3 = sum_bleu_score_3 + bleu_3
        sum_bleu_score_4 = sum_bleu_score_4 + bleu_4

        i = i + 1
        print(i)

    print('BLEU-1 : ' + str(round(sum_bleu_score_1/len(data), 4)))
    print('BLEU-2 : ' + str(round(sum_bleu_score_2/len(data), 4)))
    print('BLEU-3 : ' + str(round(sum_bleu_score_3/len(data), 4)))
    print('BLEU-4 : ' + str(round(sum_bleu_score_4/len(data), 4)))

    with open(os.environ['CURRENT_CORNELL_MODEL'] + '-' + str(sample_amount) + '.txt', 'w', encoding="latin1") as out_file:
        out_file.write('BLEU-1 : ' + str(round(sum_bleu_score_1/len(data), 4)))
        out_file.write('\n')
        out_file.write('BLEU-2 : ' + str(round(sum_bleu_score_2/len(data), 4)))
        out_file.write('\n')
        out_file.write('BLEU-3 : ' + str(round(sum_bleu_score_3/len(data), 4)))
        out_file.write('\n')
        out_file.write('BLEU-4 : ' + str(round(sum_bleu_score_4/len(data), 4)))
        out_file.write('\n')
        out_file.write('Sample amount : ' + str(sample_amount))

# CALCULATE BLEU SCORE
def bleu_score_char(dialogs, sample_amount):
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.bleu_score import SmoothingFunction
    smoothie = SmoothingFunction().method4

    if(dialogs == 'cornell'):
        dataset_folder_name = 'cornell-dialogs'
    else:
        dataset_folder_name = dialogs

    data = get_outputs_and_references(dataset_folder_name, sample_amount)
    # print('DATAAA')
    # print(data)

    sum_bleu_score_1 = 0
    sum_bleu_score_2 = 0
    sum_bleu_score_3 = 0
    sum_bleu_score_4 = 0

    i = 0

    for datum in data:
        reference = []
        reference_str = datum['reference'].lower()
        output_str = datum['output']

        reference_str = reference_str.replace(".", "")
        reference_str = reference_str.replace(",", "")
        output_str = output_str.replace(".", "")
        output_str = output_str.replace(",", "")

        reference.append(nltk.tokenize.word_tokenize(reference_str))
        candidate = nltk.tokenize.word_tokenize(output_str)
        print(reference)
        print(candidate)

        try:
            bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        except:
            bleu_1 = 0.5

        try:
            bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
        except:
            bleu_2 = 0.5

        try:
            bleu_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
        except:
            bleu_3 = 0.5

        try:
            bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        except:
            bleu_4 = 0.5

        print(bleu_1)
        print(bleu_2)
        print(bleu_3)
        print(bleu_4)

        sum_bleu_score_1 = sum_bleu_score_1 + bleu_1
        sum_bleu_score_2 = sum_bleu_score_2 + bleu_2
        sum_bleu_score_3 = sum_bleu_score_3 + bleu_3
        sum_bleu_score_4 = sum_bleu_score_4 + bleu_4

        i = i + 1
        print(i)

    print('BLEU-1 : ' + str(round(sum_bleu_score_1/len(data), 4)))
    print('BLEU-2 : ' + str(round(sum_bleu_score_2/len(data), 4)))
    print('BLEU-3 : ' + str(round(sum_bleu_score_3/len(data), 4)))
    print('BLEU-4 : ' + str(round(sum_bleu_score_4/len(data), 4)))

def interact():
    print('Mulai')
    print(os.environ['CURRENT_CORNELL_MODEL'])
    while(True):
        print('Q:')
        x = input()
        print('A : ')
        print(reply(x, 'word-glove', 'cornell')['reply'])

def interact2():
    print('Mulai')
    print(os.environ['CURRENT_CORNELL_CHAR_MODEL'])
    while(True):
        print('Q:')
        x = input()
        print('A : ')
        print(reply(x, 'char', 'cornell')['reply'])

def main():

    app.secret_key = os.urandom(12)
    try:
        port = int(os.environ['PORT'])
    except KeyError:
        print('Specify PORT as environment variable.')
        sys.exit(1)
    except TypeError:
        print('PORT must be an integer.')
        sys.exit(1)

    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    if(sys.argv[4] == 'bleu'):
        if(dataset == 'cornell'):
            if(type == 'char'):
                bleu_score('cornell', 2000/2)
            else:
                bleu_score('cornell', 2000/2)
        elif(dataset == 'gunthercox'):
            bleu_score('gunthercox', 10)
        elif(dataset == 'twitter'):
            bleu_score('twitter', 200)
    elif(sys.argv[4] == 'test'):
        main()
    elif(sys.argv[4] == 'terminal'):
        interact()
    elif(sys.argv[4] == 'terminal2'):
        interact2()
