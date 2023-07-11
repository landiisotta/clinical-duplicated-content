"""
Module for clinical text tokenization with custom sentencizer and tokenizer.
Sentences: Text starting with capital letter and ending with period;
Tokens: words, lab results.
Dates and times are replaced with special tokens [DATE] and [TIME], respectively.
"""

from spacy.language import Language
from spacy.tokens import Token
import spacy
import re


# Custom sentence tokenizer for clinical notes
@Language.component("custom_sentencizer")
def custom_sentencizer(doc):
    """
    Custom sentencizer.
    :param doc: Document
    :type doc: spacy.tokens.doc.Doc
    :return: tokenized document
    :rtype: spacy.tokens.doc.Doc
    """
    for i, token in enumerate(doc[:-1]):
        # Full stop for sentence tokenization
        if re.match(r'^ ?\.', token.text) and (doc[i + 1].is_upper or doc[i + 1].is_title):
            doc[i + 1].is_sent_start = True
            token.is_sent_start = False
        # Numeric list for sentence tokenization
        elif re.match(r'[0-9]{1,2}\.$', token.text):
            if not doc[i - 1].is_stop:
                token.is_sent_start = True
                doc[i + 1].is_sent_start = False
                token._.is_list = True
            else:
                token.is_sent_start = False
                doc[i + 1].is_sent_start = True
        # Bullet point list for sentence tokenization
        elif token.text == '-' and doc[i + 1].text != '-':
            token.is_sent_start = True
            doc[i + 1].is_sent_start = False
            token._.is_list = True
        else:
            doc[i + 1].is_sent_start = False
    return doc


# Custom tokenizer
@Language.component('tkndef')
def def_tokens(doc):
    """
    Custom tokenizer for special tokens, e.g., abbreviations
    :param doc: document
    :type doc: spacy.tokens.doc.Doc
    :return: tokenized document
    :rtype: spacy.tokens.doc.Doc
    """
    patterns = [r'\[\*\*.+?\*\*\]',  # de-identification
                r'[0-9]{1,4}[/\-]([0-9]{1,2}|[a-zA-Z]{3})[/\-]*[0-9]*$',  # date
                r'([Jj]an(uary)?|[Ff]eb(ruary)?|[Mm]ar(ch)?]|[Aa]pr(il)?]|'
                r'[Mm]ay|[Jj]une?|[Jj]uly?|[Aa]ug(ust)?|[Ss]ep(tember)?|'
                r'[Oo]ct(ober)?|[Nn]ov(ember)?|[Dd]ec(ember)?) '
                r'[0-9]{1,2} ?,? [0-9]{2,4}',  # extended date
                r'[0-9]*\-?[0-9]+%?',  # lab/test result
                r'[0-9]+/[0-9]+',  # lab/test result
                r'[0-9]{0,2}\.[0-9]{1,2}',  # lab/test result
                # r'([0-9]{1,3} ?, ?[0-9]{3})+',  # number >= 10^3
                r'[0-9]{1,2}\+',  # lab/test result
                r'[A-Za-z]{1,3}\.',  # abbrv, e.g., pt.
                r'[A-Za-z]\.([A-Za-z]\.){1,2}',  # abbrv, e.g., p.o., b.i.d.
                r'[0-9]{1,2}h\.',  # time, e.g., 12h
                r'(\+[0-9] ?)?\(?[0-9]{3}\)?[\- ][0-9]{2,4}[\- ][0-9]{2,4}(-[0-9]{1})?',
                # phone number and discharge order
                r'[0-9]{1,2}\.',  # Numbered lists
                r'([0-9]{1,2}:*[0-9]{0,2}:*[0-9]{0,2}( AM| PM)|[0-9]{1,2}:[0-9]{2}$)',  # times
                r'([A-Za-z0-9]+\-)+[A-Za-z0-9]+',  # dashed words
                r'[0-9]+s',  # decades
                r'q\.[0-9]h',  # every x hours
                r'([0-9]{1,2} ?,? )[0]{3} (mg|cc|MG|CC)\.?',  # cases like  1 , 000 MG
                r'[0-9]{1,3}[-\.]*[0-9]* *(mg|cc|MG|CC)\.?'
                # r'[A-Za-z0-9]+'  # Chemical bounds
                ]
    for expression in patterns:
        for match in re.finditer(expression, doc.text):
            start, end = match.span()
            span = doc.char_span(start, end)
            # This is a Span object or None if match
            # doesn't map to valid token sequence
            if span is not None:
                with doc.retokenize() as retokenizer:
                    retokenizer.merge(span, attrs={"IS_ALPHA": True})
    return doc


nlp = spacy.load('en_core_sci_md-0.5.1/en_core_sci_md/en_core_sci_md-0.5.1/',
                 disable=["ner"])
nlp.add_pipe('tkndef', before='parser')
nlp.add_pipe('custom_sentencizer', before='parser')
Token.set_extension('is_list', default=False, force=True)


def tokenize(x):
    note = nlp(re.sub('  +', ' ', x.replace('\n', ' ')))
    tkn_note = ' . '.join(list(map(sent_tokenize, note.sents)))
    return tkn_note


def sent_tokenize(sen):
    tkn = []
    for t in sen:
        if not t._.is_list:
            # replace dates with special token
            if re.match(r'([0-9]{1,2}|[0-9]{4})[\/\-]([0-9]{1,2}|[a-zA-Z]{3})[\/\-]([0-9]{1,2}|[0-9]{4})$',
                        t.text) or re.match(r'([Jj]an(uary)?|[Ff]eb(ruary)?|[Mm]ar(ch)?]|[Aa]pr(il)?]|'
                                            r'[Mm]ay|[Jj]une?|[Jj]uly?|[Aa]ug(ust)?|[Ss]ep(tember)?|'
                                            r'[Oo]ct(ober)?|[Nn]ov(ember)?|[Dd]ec(ember)?) '
                                            r'[0-9]{1,2} ?,? [0-9]{2,4}', t.text):
                tkn.append('[DATE]')
            # replace time with special tokens
            elif re.match(r'([0-9]{1,2}:*[0-9]{0,2}:*[0-9]{0,2}( AM| PM)|[0-9]{1,2}:[0-9]{2}$)', t.text):
                tkn.append('[TIME]')
            # remove anonymized tokens
            elif re.match(r'\[\*\*.+?\*\*\]', t.text):
                continue
            # remove phone numbers
            elif re.match(r'(\+[0-9] ?)?\(?[0-9]{3}\)?[\- ][0-9]{2,4}[\- ][0-9]{2,4}(-[0-9])?', t.text):
                continue
            # remove mrn-like numbers
            elif re.match(r'[0-9]{4}[0-9]+', t.text):
                continue
            # replace 10,000 --> 10000 to avoid weird tokenization during fine-tuning
            elif re.match(r'[0-9]{1,3},[0-9]{3}', t.text):
                tkn.append(re.sub(',', '', t.text))
            # Notes that have >=10^3 numbers with intermediate spaces and special case of ABG reportings
            # where three different values are reported
            elif re.match(r'[0-9]{1,3} , [0-9]{3}', t.text) and 'ABG' not in tkn:
                tkn.append(re.sub(' ', '', re.sub(' , ', '', t.text)))
            elif not t.is_alpha:
                continue
            else:
                tkn.append(re.sub(' ', '', t.text.strip('.')))
    if len(tkn) > 0:
        return ' '.join(tkn).strip(' ')
    else:
        return ''
