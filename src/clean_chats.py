import os
import json
import itertools
import re
import unicodedata
from datetime import date

import pickle


def create_user_reference(discord_json):
    user_ids = {id:discord_json['meta']['users'][id]['name'] for id in dis_json['meta']['users']}
    user_index = discord_json['meta']['userindex']
    user_ref = {user_index.index(id):user_ids[id] for id in user_index}
    return(user_ref)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def clean_string(s, remove_punc = False):
    s = re.sub(r"([\<]).*?([\>])", "", s).strip()

    if remove_punc:
        s = re.sub(r"([.!?])", r" ", s)
    else:
        s = re.sub(r"([.!?])", r" \1", s)

    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = s.lower()
    return s

if __name__ == "__main__":
    remove_punc = True
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'bin', 'chat_data')

    # Load Discord JSON
    with open(os.path.join(data_dir, 'dht_20200319.txt')) as h:
        dis_json = json.load(h)

    # Create user index reference
    user_ref = create_user_reference(dis_json)

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
    bots = [6,7,8,11,12,13,14,15,17,18,19,20]

    for message in chats:
        single_msg = message
        
        # Skip messages with embeds or links
        if ('e' in single_msg or 'a' in single_msg):
            continue

        # Skip messages with no message
        if 'm' not in single_msg:
            continue

        # Skip messages with with no Text
        if not single_msg['m']:
            continue
            
        # Skip Bracketed Convos
        if (single_msg['m'] and 
            single_msg['m'][0] in ['(','[','{']):
            continue
            
        # Skip messages from quotebot 
        # if user_ref[single_msg['u']] == 'quote-bot':
        #     continue

        # Skip messages from bots
        if single_msg['u'] in bots:
            continue

        # Clean Message String
        single_msg['m'] = clean_string(single_msg['m'], remove_punc=remove_punc)

        # # Convert to ASCII
        #single_msg['m'] = unicode_to_ascii(single_msg['m'])

        # Parse Others' Messages
        if user_ref[single_msg['u']] != "synsypa":
            # Message and Response present and ready to be written
            # This should only trigger if the last speaker was response
            if (need_resp and resp_ready 
                and k_time < o_time + (2 * 60 * 1000)):
                responses.append((' '.join(o_msg).strip(), ' '.join(k_msg).strip()))
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
                responses.append((' '.join(o_msg).strip(), ' '.join(k_msg).strip()))
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
        responses.append((' '.join(o_msg).strip(), ' '.join(k_msg).strip()))

    pickle.dump(responses, open(os.path.join(data_dir, f'/clean_conversations_{date.today()}.pkl'), 'wb'))