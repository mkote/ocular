def scoring(predicted, labels):
    correct = 0
    incorrect = 0
    wrong_list = []
    for i, pair in enumerate(zip(predicted, labels)):
        if pair[0] == int(pair[1]):
            correct += 1
        else:
            incorrect += 1
            wrong_list.append(i)

    score = (float(correct) / float(len(predicted))) * 100
    return score, wrong_list
