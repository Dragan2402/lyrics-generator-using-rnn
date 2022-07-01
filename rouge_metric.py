from datasets import load_metric
from os import system

system('cls')
rouge = load_metric("rouge")
print("Rouge metric loaded\n")
result = open("results/taylor_2.txt", encoding='UTF-8').read()
all_references = []
raw_result = result.lower()
result_length = len(raw_result)
print("Result song loaded\n")
print("Size of loaded song(char):" + str(result_length) + "\n")
all_references.append(raw_result)

raw_text_validate = open("data/lyricsText_validation.txt", encoding='UTF-8').read()
print("Validation songs loaded\n")
raw_text_validate = raw_text_validate.lower()
# print("Chopping validation song into pieces\n")
# chunks = [raw_text_validate[i:i + result_length] for i in range(0, len(raw_text_validate), result_length)]
scores = []
all_hypothesis = [raw_text_validate]
result = rouge.compute(predictions=all_references, references=all_hypothesis, use_aggregator=True)
result_rouge1 = float(result["rouge1"].low.precision) * 100
formated_result = "{:.2f}".format(result_rouge1)
scores.append(formated_result)

# print("\n\nScores for validation songs are:\n")
# score_sum = 0
# for score in scores:
#     score_sum += float(score)
#     print(str(score) + "%")
print("\n\nRouge metric matching score for generated song: " + str(max(scores)) + "%")
# average_score = score_sum / float(len(scores))
# formated_result = "{:.2f}".format(average_score)
# print("Average score is: " + str(formated_result) + "%")
