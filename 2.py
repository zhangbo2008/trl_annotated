import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
model = AutoModelForCausalLM.from_pretrained('huggyllama/llama-7b', device_map='auto')

enc = lambda s: tokenizer.encode(s, add_special_tokens=False)

def test(question_enc, answer_enc):
    # get b logprobs out of model(a + b)
    logprobs = F.log_softmax(model(torch.tensor(question_enc + answer_enc).unsqueeze(0).cuda())['logits'], dim=-1).cpu().squeeze(0)[len(question_enc) - 1 : -1]
    guess = logprobs.argmax(dim=-1)
    print(f'|{tokenizer.decode(guess)}| |{tokenizer.decode(answer_enc)}|')

# assume enc(a + b) = enc(a) + enc(b)

test(enc('Paris is the capital of'), enc(' France.'))
# |France2,| | France.|

# assume enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]

sentence_enc = enc('Paris is the capital of France.')
question_enc = enc('Paris is the capital of')
answer_enc = sentence_enc[len(question_enc):]

test(question_enc, answer_enc)
# |France.| |France.|