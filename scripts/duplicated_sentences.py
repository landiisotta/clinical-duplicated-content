from collections import Counter
import tqdm
import argparse
import re
import numpy as np

"""
This script extracts duplicated sentences from duplicated text chunks
as the text 
- stripped of unicode characters;
- that starts with a capital letter (or corresponds to the beginning of a note) and ends with ' . ';
- is repeated at least twice;
- is longer than 5 characters.
Then it separates byte spans that correspond to within-note duplications
(marking only the second occurrence) and duplicated text throughout the corpus.

Input: (1) dataset.split; (2) dataset.split.remove.byterange

Output: (1) dataset.split.remove.byterange.wnr; (2) dataset.split.remove.byterange.bysen
"""

parser = argparse.ArgumentParser(description='Detect duplicated sentences.')
parser.add_argument('--dataset', type=str)
parser.add_argument('--remove_range', type=str)

config = parser.parse_args()


def check_span(string):
    """Returns the index after multiple blanks"""
    string_split = string.split(b' ')
    index = 0
    try:
        while string_split[index] == b'':
            index += 1
    except IndexError:
        index -= 1
    return index


def filter_sentences(text, b_tuple):
    """ Select sentences that start with a capital letter or a number or have a unicode character at the beginning.
    Drop sentences with length <= 5 characters or that do not qualify as sentences."""
    b_start, b_end = b_tuple[0], b_tuple[1]

    match_search = list(re.finditer(b'\x00', text))
    if len(match_search) > 0:
        for s in match_search:
            text_tmp = text[s.span()[1]:]
            b_start_tmp = b_start + s.span()[1]
        text, b_start = text_tmp, b_start_tmp

    m = re.search(br'[A-Z]|[0-9]|\[[A-Z]+\]', text)
    if m:
        text = text[m.span()[0]:]
        b_start += m.span()[0]
    # if re.search(b'LDL in this case is less than 70 Umberto Uzzell MD Signed electronically by Umberto Uzzell', text):
    #     print(text, b_tuple, b_start)
    if len(text) > 5:
        return text, (b_start, b_end)
    else:
        return None


# Read dataset (utf-8 encoded)
dataset = open(config.dataset, 'rb').read()
print(f"Loaded: {config.dataset}\n")
# Read byte ranges corresponding to duplicated content
dup_text = []  # Text saved as utf-8 encoded
with open(config.remove_range, 'r') as f:
    for line in f:
        if 'out' in line:
            break
    for line in f:
        idx = list(map(int, line.strip().split()))
        dup_text.append((dataset[idx[0]:idx[1]], tuple(idx)))
# Extract note ranges: list byte ranges corresponding to each note
sizes = np.frombuffer(open(config.dataset + "." + "size", "rb").read(),
                      dtype=np.uint64)
note_span = [[s, sizes[i + 1]] for i, s in enumerate(sizes[:-1])]
# UID = 1
# for idx, s in enumerate(re.finditer(b'\xff\xff', dataset)):
#     if b'\xff\xff' + struct.pack("<I", UID) == dataset[s.span()[0]:s.span()[0] + 6]:
#         UID += 1
#         if idx == 0:
#             note_span.append([s.span()[0], None])
#         else:
#             note_span[-1][1] = s.span()[0]
#             note_span.append([s.span()[0], None])
#     else:
#         continue
# note_span[-1][1] = len(dataset)
# assert UID - 1 == len(note_span)

# Extract duplicated sentences
dup_sen = []
for el in tqdm.tqdm(dup_text, total=len(dup_text), desc="Extracting duplicated sentences"):
    dup_sen_tmp = []
    # Split on full stop to extract single sentences
    v_sen = el[0].split(b' . ')
    for ch, sen in enumerate(v_sen):
        if ch == 0:
            start_next = el[1][0]
        else:
            # start_next = dup_sen_tmp[-1][1][1] + 3
            start_next = old_byte_end + 3
        # Get rid of blanks
        if re.match(b' ', sen):
            i = check_span(sen)
            byte_start = start_next + i
        else:
            byte_start = start_next
        new_sen = sen.strip(b' ')
        byte_end = byte_start + len(new_sen)
        if len(new_sen) < len(sen) and not re.match(b' ', sen):
            old_byte_end = byte_start + len(sen)
        else:
            old_byte_end = byte_end
        new_sen_tup = (new_sen, (byte_start, byte_end))
        # if re.search(b'The cardiac catheterization results at that time revealed a right atrial pressure of 7', el[0]):
        #     print(new_sen_tup, dataset[byte_start:byte_end], start_next, el[1][0], byte_start, byte_end, old_byte_end, el[0])
        #     print(dataset[el[1][0]:el[1][0]+len(v_sen[0])])
        #     print(dataset[el[1][0]+len(v_sen[0])+3:el[1][0]+len(v_sen[0])+3+len(v_sen[1])])
        #     print(new_sen_tup, sen.strip(b' '), new_sen)
        #     print(byte_start, el[1][0]+len(v_sen[0])+3)
        #     print(dup_sen_tmp)
        #     print(byte_end, old_byte_end)
        # Get rid of unicode characters that separate notes
        if re.match(b'\xff\xff', new_sen):
            if len(new_sen) > 6:
                new_sen = new_sen[6:]
                byte_start += 6
            new_sen_tup = (new_sen, (byte_start, byte_end))
        s = re.search(b'\xff\xff', new_sen)
        new_sen_v = []
        if s:
            v = new_sen.split(b'\xff\xff')
            new_sen_v = [(v[0], (byte_start, byte_start + len(v[0])))]
            for sentence in v[1:]:
                if len(sentence) >= 4:
                    split_sen = sentence[4:]
                    split_byte_start = new_sen_v[-1][1][1] + 6
                else:
                    split_sen = sentence
                    split_byte_start = new_sen_v[-1][1][1] + 2
                split_byte_end = split_byte_start + len(split_sen)
                new_sen_v.append((split_sen, (split_byte_start, split_byte_end)))
            dup_sen_tmp.extend(new_sen_v)
        else:
            dup_sen_tmp.append(new_sen_tup)
        # Filter out sentences with less than 5 characters and text that does not qualify
        # as sentence
    for ns in dup_sen_tmp:
        filtered_sen = filter_sentences(ns[0], ns[1])
        if filtered_sen:
            try:
                filtered_sen[0].decode('utf8')
            except UnicodeDecodeError:
                print(f"Unicode character still present in {filtered_sen[0]}")
                print(f'Old: {filtered_sen}')
                # print(ns[0][:-1], (ns[1][0], ns[1][1] - 1))
                # filtered_sen = (ns[0][:-1], (ns[1][0], ns[1][1] - 1))
                filtered_sen = (filtered_sen[0][:-1], (filtered_sen[1][0], filtered_sen[1][1] - 1))
                print(f'New: {filtered_sen}')
            assert filtered_sen[0] == dataset[filtered_sen[1][0]:filtered_sen[1][1]]
            # try:
            #     assert filtered_sen[0] == dataset[filtered_sen[1][0]:filtered_sen[1][1]]
            # except AssertionError:
            #     print(filtered_sen[0])
            #     print(dataset[filtered_sen[1][0]:filtered_sen[1][1]])
            #     print(start_next, byte_start, old_byte_end, byte_end)
            #     print(filtered_sen[1][0],filtered_sen[1][1])
            #     for el in dup_sen_tmp:
            #         print(el[0], el[1][0], el[1][1])
            #     raise AssertionError
            dup_sen.append(filtered_sen)

# dup_sen = []
# for el in dup_text:
#     sents = el[0].split(b' . ')
#     for ch, sen in enumerate(sents):
#         if ch == 0:
#             dup_sen.append((sen, (el[1][0], el[1][0] + len(sen))))
#         else:
#             # +3 because we split on ' . ' (3 characters)
#             start_next = dup_sen[-1][1][1]
#             dup_sen.append((sen, (start_next + 3, start_next + 3 + len(sen))))
#         assert sen == dataset[dup_sen[-1][1][0]:dup_sen[-1][1][1]]

# Select sentences that start with a capital letter (i.e., complete sentences)
# that are longer than 5 characters
# and that occur more than once
dup_sen_count = Counter([el[0] for el in dup_sen])
dup_sen_count = {sen: count for sen, count in dup_sen_count.items() if count > 1}
print(f"Found {len(dup_sen_count)} unique duplicated sentences.")
# Duplicated sentences with byte range
dup_sen_rid = [el for el in dup_sen if el[0] in dup_sen_count]
print(f"Total duplicated sentences {len(dup_sen_rid)}")

# Create dictionary with sen: [index tuples]
# Create set with index tuples
dup_sen_span = {}
range_vec = set()
for el in dup_sen_rid:
    dup_sen_span.setdefault(el[0], list()).append(el[1])
    range_vec.add(el[1][0])

# Create a lookup table where each character position
# that corresponds to the beginning of a duplicated sentence
# is associated to the note id number
lookup_sen = {}
for idx, el in tqdm.tqdm(enumerate(note_span), total=len(note_span), desc='Building lookup table'):
    for i in range(el[0], el[1]):
        if i in range_vec:
            lookup_sen[i] = idx

# Within-note redundancy
wnr = []
bnr = []
for sen, s_vec in tqdm.tqdm(dup_sen_span.items(), desc='Extracting within-note redundancy'):
    i = 0
    while i < len(s_vec) - 1:
        if i == 0:
            bnr.append(s_vec[i])
        for j in range(i + 1, len(s_vec)):
            if lookup_sen[s_vec[i][0]] == lookup_sen[s_vec[j][0]]:
                wnr.append(s_vec[j])
                continue
            else:
                bnr.append(s_vec[j])
                break
        i = j

assert len(bnr) == len(dup_sen_rid) - len(wnr)
bnr = sorted(bnr)
wnr = sorted(wnr)
# Save byte ranges for within-note duplicated sentences
with open(config.remove_range + '.wnr', 'w') as f:
    f.write('out\n')
    for br in wnr:
        f.write(str(br[0]) + ' ' + str(br[1]) + '\n')

# Save new byterange file
with open(config.remove_range + '.bysen', 'w') as f:
    f.write('out\n')
    for el in bnr:
        f.write(str(el[0]) + ' ' + str(el[1]) + '\n')
