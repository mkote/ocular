from numpy import array, bincount, argmax


def csp_voting(band_list):
    band_zip = zip(*[band_list[x] for x in range(0, len(band_list))])
    voting_results = []
    for trial in band_zip:
        combined_bands = array(trial).ravel().tolist()
        trial_result = []

        for word in combined_bands:
            char_list = [int(x) for x in str(word)]

            for char in char_list:
                trial_result.append(int(char))

        voting_results.append(argmax(bincount(trial_result)))

    return voting_results
