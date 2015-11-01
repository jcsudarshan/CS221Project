import util_baseline

def windowPredictor(raw_in, features, weights):

    #---Params:
    use_punc = False

    window_size = 4 #Best is 4
    flip_thresh = 0.7 #Best is 0.4
    up_thresh = 0.3
    down_thresh = 0
    upscale_amount = 2.5 #Best is 2.5
    downscale_amount = 0 #Best is 0
    flip_amount = -1 #Best is 1
    #---

    w = weights.copy()

    negs = ["hardly", "neither", "no", "nobody","not","cannot","didn't","haven't","never","lacking","lack","wasn't","isn't","can't","won't","doesn't","hasn't","without"] #ISSUE: Should use stemming.
    pivots = ["but", "however"]
    amps = ['much', 'very']
    puncs = ['.', ',']

    raw_list = raw_in.split()

    neg_positions = [raw_list.index(neg) for neg in raw_list if neg in negs]
    pivot_positions = [raw_list.index(pivot) for pivot in raw_list if pivot in pivots]
    amp_positions = [raw_list.index(amp) for amp in raw_list if amp in amps]
    punc_positions = [raw_list.index(punc) for punc in raw_list if punc in puncs]

    change_words = set()

    flip_words = set()
    upscale_words = set()
    downscale_words = set()

    for pos in neg_positions:
        if use_punc:
            for p in punc_positions:
                if p > pos:
                    break

            segment = raw_list[pos+1:pos+1+window_size]
        else:
            segment = raw_list[pos+1:pos+1+window_size]

        for word in segment:
            change_words.add(word)
            flip_words.add(word)

    for pos in amp_positions:
        if use_punc:
            for p in punc_positions:
                if p > pos:
                    break

            segment = raw_list[pos+1:p]
        else:
            segment = raw_list[pos+1:pos+1+window_size]

        for word in segment:
            change_words.add(word)
            upscale_words.add(word)

    for pos in pivot_positions:
        if use_punc:
            for p in punc_positions:
                if p > pos:
                    break

            up_segment = raw_list[pos+1:p] #How to treat down case?

        else:
            up_segment = raw_list[pos+1:pos+1+window_size]

        down_segment = raw_list[pos-window_size:pos]

        for word in down_segment:
            #print(raw_in)
            change_words.add(word)
            downscale_words.add(word)

        for word in up_segment:
            change_words.add(word)
            upscale_words.add(word)

    for word in change_words:
        try: #ISSUE: What should we do if there are unseen words?
                if word in flip_words and abs(w[word]) >= flip_thresh:
                    w[word] = w[word]*(flip_amount)

                if word in upscale_words and abs(w[word]) >= up_thresh:
                    w[word] = w[word]*(upscale_amount)

                if word in downscale_words and abs(w[word]) >= down_thresh:
                    w[word] = w[word]*(downscale_amount)
        except KeyError:
            pass

    return (1 if util_baseline.dotProduct(features, w) >= 0 else -1)
