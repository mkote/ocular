def scoring(predicted, labels):
    correct = 0
    incorrect = 0
    for pair in zip(predicted, labels):
        if pair[0] == int(pair[1]):
            correct += 1
        else:
            incorrect += 1

    score = (float(correct) / float(len(predicted))) * 100
    return score