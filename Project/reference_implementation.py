from model import create_dataloader, Net
import torch
from gensim.models import KeyedVectors
import torch.optim as optim
from torch.autograd import Variable
import argparse
import torchvision.models as models
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, required=False)
args = parser.parse_args()


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LR = 1e-3

def hook_fn(module, input, output):
    setattr(module, "_value_hook", output)

def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

class FilterVisualizer():
    def __init__(self, vocab_size, pretrained_embeddings, device, model):
        # init the network
        self.model = model.eval()
        set_trainable(self.model, False)
    
    def visualize(self, layer_name, vocab_size, words_to_idx, optim_steps=20):
        # create input to "train"
        print(words_to_idx)
        random_sentence = torch.LongTensor(1, 1).random_(0, vocab_size)
        for k, v in words_to_idx.items():
            if random_sentence[0].item() == v:
                print("random word {}".format(k))
                break
        random_sentence = model.embeds_input(random_sentence)
        random_sentence = Variable(random_sentence, requires_grad=True)

        # hook the layer we want to hook
        for n, m in self.model.named_modules():
            if n == layer_name:
                print("Register hook for layer named {}".format(layer_name))
                m.register_forward_hook(hook_fn)
                break

        optimizer = torch.optim.Adam([random_sentence], lr=LR)
        for nm in range(optim_steps):
            optimizer.zero_grad()
            self.model(random_sentence, embedded_input=True)
            for n, m in self.model.named_modules():
                if n == layer_name:
                    loss = -m._value_hook.mean()
                    print("loss val: {:.3f}".format(loss))
                    loss.backward()
                    optimizer.step()
            

        t = model.get_embedding()
        # https://discuss.pytorch.org/t/vec2word-or-something-similar/2068/2
        normalized_embedding = t.weight/((t.weight**2).sum(0)**0.5).expand_as(t.weight)
        similarity, words = torch.topk(torch.mv(normalized_embedding, random_sentence.squeeze(0).squeeze(0)), 5)
        print(similarity, words)
        for i, word in enumerate(words):
            for k, v in words_to_idx.items():
                if word.item() == v:
                    print("{}:th closest word {}".format(i, k))
                    break



        t = model.get_embedding()
        # https://discuss.pytorch.org/t/vec2word-or-something-similar/2068/2
        normalized_embedding = t.weight/((t.weight**2).sum(0)**0.5).expand_as(t.weight)
        similarity, words = torch.topk(torch.mv(normalized_embedding, random_sentence.squeeze(0).squeeze(0)), 5)
        print(similarity, words)
        for i, word in enumerate(words):
            for k, v in words_to_idx.items():
                if word.item() == v:
                    print("{}:th closest word {}".format(i, k))
                    break



if __name__ == "__main__":
    print("device to use: {}".format(DEVICE))

    # load pretrained model from file
    pretrained_embeddings = KeyedVectors.load("models/word2vec_m5_s300_w8_s0_h0_n5_i10")

    train_loader, val_loader, test_loader, vocab_size, val_labels, words_to_idx = create_dataloader("data/train.csv")

    model = Net(vocab_size=vocab_size, pretrained_embeddings=pretrained_embeddings, device=DEVICE)

    # train the network
    if args.load is None:
        best_model = model._train(train_loader, val_loader, test_loader, val_labels, save=True)
    else:
        best_model = model.load(args.load)

    f = FilterVisualizer(vocab_size=vocab_size, pretrained_embeddings=pretrained_embeddings, device=DEVICE, model=best_model)
    f.visualize("fc2", vocab_size, words_to_idx, optim_steps=10)
   
