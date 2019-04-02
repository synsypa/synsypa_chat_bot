import json
import itertools
import re
import csv
import numpy as np

# Load Discord JSON
with open('chat_data/discord_20190318.txt') as h:
    dis_json = json.load(h)

# Create user index reference
user_ids = {id:dis_json['meta']['users'][id]['name'] for id in dis_json['meta']['users']}
user_index = dis_json['meta']['userindex']
user_ref = {user_index.index(id):user_ids[id] for id in user_index}

# Sort Chats by timestamp (ascending)
chats = [msg for mid, msg in sorted(dis_json['data']['106536439809859584'].items())]

# Clean and Parse Message and response
# Initialize
responses = []
messages_list = []
k_msg, o_msg = [], []
k_time, o_time = None, None
o_speak = None
need_resp = False
resp_ready = False

for message in chats:
    single_msg = message
    # Remove Lines with no Text
    if not single_msg['m']:
        continue
    # Remove Bracketed Convos
    if (single_msg['m'] and 
        single_msg['m'][0] in ['(','[','{']):
        continue
    # Remove Messages with embeds or links
    if ('e' in single_msg or 'a' in single_msg):
        continue
    if user_ref[single_msg['u']] == 'quote-bot':
        continue
        
    # Clean out custom emoji
    single_msg['m'] = re.sub("([\<]).*?([\>])", "", single_msg['m']).strip()
    # Remove non-terminating punctuation
    single_msg['m'] = re.sub(r"[^a-zA-Z0-9.!?\s]+", "", single_msg['m'])
    # Separate Punctuation
    single_msg['m'] = re.sub(r"([.!?])", r" \1", single_msg['m'])
    # Make lower case
    single_msg['m'] = single_msg['m'].lower()
    
    # Parse Others' Messages
    if user_ref[single_msg['u']] != "synsypa":
        # Message and Response present and ready to be written
        # This should only trigger if the last speaker was response
        if (need_resp and resp_ready 
            and k_time < o_time + (2 * 60 * 1000)):
            responses.append((' '.join(o_msg), ' '.join(k_msg)))
            k_msg, o_msg = [], []
            k_time, o_time = None, None
            o_speak = None
            need_resp = False
            resp_ready = False
        # First message 
        if not o_msg:
            o_speak = single_msg['u']
            o_msg.append(single_msg['m'])
            o_time = single_msg['t']
            need_resp = True
        # Same speaker, within 2 mins
        elif (o_time > single_msg['t'] - (2 * 60 * 1000) and
                 single_msg['u'] == o_speak):
            o_msg.append(single_msg['m'])
            o_time = single_msg['t']
        # Different non-response speaker or outside of 2 mins
        # Reset message tracking
        else:
            o_msg = [single_msg['m']]
            o_time = single_msg['t']
            o_speak = single_msg['u']
            k_msg = []
            k_time = None
            resp_ready = False
    
    # Parse responses
    if user_ref[single_msg['u']] == "synsypa":
        # First response
        if not k_msg:
            k_msg.append(single_msg['m'])
            k_time = single_msg['t']
            # If a message needs a response, flag response available
            if need_resp:
                resp_ready = True
        # Response in the queue and next is response w/i 2 mins
        elif k_time > single_msg['t'] - (2 * 60 * 1000):
            k_msg.append(single_msg['m'])
            k_time = single_msg['t']
        # Message and response ready, but incoming message is not w/i 2 mins
        elif (need_resp and resp_ready 
            and k_time < o_time + (2 * 60 * 1000)):
            responses.append((' '.join(o_msg), ' '.join(k_msg)))
            k_msg, o_msg = [], []
            k_time, o_time = None, None
            o_speak = None
            need_resp = False
            resp_ready = False
        else: 
            k_msg, o_msg = [], []
            k_time, o_time = None, None
            o_speak = None
            need_resp = False
            resp_ready = False

# Write the leftover message and response if both are ready
if need_resp and resp_ready and k_time < o_time + (5 * 60 * 1000):
    responses.append((' '.join(o_msg), ' '.join(k_msg)))
    
# Output message,response as csv
# with open('conversations.csv', 'w') as c:
#    conv_writer = csv.writer(c)
#    for row in responses:
#        conv_writer.writerow(row)
np.save('chat_data/clean_conversations.npy', responses)

# Output messages only (Deprecated for Torch Version)
#with open('chat_data/clean_messages.txt', 'w') as m:
#    for msg, resp in responses:
#        m.write(msg + ' ' + resp + ' ')