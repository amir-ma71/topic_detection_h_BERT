from flask import Flask, request
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from langdetect import detect, DetectorFactory


def prepare_input(sent):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    tokenizer = AutoTokenizer.from_pretrained("./src/heBERT")

    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=64,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    input_ids = encoded_dict['input_ids']

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks = encoded_dict['attention_mask']

    # input_ids = torch.cat(input_ids, dim=0)
    # attention_masks = torch.cat(attention_masks, dim=0)

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler)

    return prediction_dataloader


def percent(arrey):
    min_num = -min(arrey)
    new = []
    for n in arrey:
        new.append(n + min_num + 0.01)

    sum_num = sum(new)
    percen = []
    for m in new:
        percen.append((100 * m) / sum_num)

    return percen


model = torch.load("./src/Output_models/Bert_model.model", map_location=torch.device('cpu'))
model = model.to("cpu")
model.eval()

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/predict', methods=['POST'])
def prepare_text():
    # Catch the image file from a POST request
    # if 'file' not in request.files:
    #     return "Please try again. The Image doesn't exist"

    request_data = request.get_json()

    sent = request_data['text']

    # Detect language of text
    try:
        DetectorFactory.seed = 0
        lang = detect(sent)
    except:
        return "No text enter",400

    if lang != "he":
        return "the language is not Hebrew, please type Hebrew language to detect topic.",400

    if not sent:
        return

    if len(sent)< 85:
        return "your text is too small, please type more words to detect",400
    # Prepare the text
    prediction_dataloader = prepare_input(sent)

    b_input_ids, b_input_mask = prediction_dataloader.dataset.tensors

    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.numpy()

    pred_percent = percent(logits[0])

    label_list = ["entertainment", "sport", "sociopolitics", "technology", "health", "economy"]
    final_dict = {"probability": {},
                  "dependency": []}

    for i in range(len(label_list)):
        final_dict["probability"][label_list[i]] = float(round(pred_percent[i], 2))

    final_dict["probability"] = {k: v for k, v in
                                 sorted(final_dict["probability"].items(), key=lambda item: item[1], reverse=True)}
    final_dict["dependency"].append(list(final_dict["probability"].keys())[0])
    if list(final_dict["probability"].values())[1] >= 30:
        final_dict["dependency"].append(list(final_dict["probability"].keys())[1])



    # Return on a JSON format
    return final_dict


@app.route('/check', methods=['GET'])
def check():
    return "every things right! "

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
