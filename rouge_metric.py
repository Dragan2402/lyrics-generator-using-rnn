from datasets import load_metric

raw_text_validate = open("data/lyricsText_validation.txt", encoding='UTF-8').read()
raw_text_validate = raw_text_validate.lower()
validate_songs = raw_text_validate.split("\n\n")
all_hypothesis = []
# for song in validate_songs:
#     lines = song.split("\n")
#     for line in lines:
#         all_hypothesis.append(line)
all_hypothesis.append(raw_text_validate)
result = open("results/taylor.txt", encoding='UTF-8').read()
all_references = []
raw_result = result.lower()
lines_result = raw_result.split("\n")
# for line in lines_result:
#     all_references.append(line)
all_references.append(raw_result)
rouge = load_metric("rouge")
result = rouge.compute(predictions=all_hypothesis,references= all_references)
print(result)
list_results= result["rougeL"]
for result_ in list_results:
    print(result_)