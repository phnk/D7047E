from model import create_dataloader, Net
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
import torch.optim as optim
from torch.autograd import Variable
import argparse
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import re

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, required=False)
args = parser.parse_args()
scats = []


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LR = 1e-3

def hook_fn(module, input, output):
    setattr(module, "_value_hook", output)

def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def update_plot(i, static_x, static_y, moving_x, moving_y, ax, labels):
    global scats
    for scat in scats:
        scat.remove()

    scats = []
    if i > 0:
        static_x.pop()
        static_y.pop()
    
        static_x.append(moving_x[i])
        static_y.append(moving_y[i])

    for j in range(len(static_x)):
        scats.append(ax.scatter(static_x[j], static_y[j]))

def save_embeddings(model, res, words_to_idx, visualisation_model):
    labels = []
    tokens = []
    i = 0
    t = model.get_embedding()

    for word in res["closest_words_before_training"]:
        tensor = torch.LongTensor([[words_to_idx[word]]])
        tensor = model.embeds_input(tensor).clone()
        tensor = tensor.squeeze(0).squeeze(0).data
        tokens.append(np.array(tensor))
        labels.append(word)
 
    for word in res["closest_words_after_training"]:
        tensor = torch.LongTensor([[words_to_idx[word]]])
        tensor = model.embeds_input(tensor).clone()
        tensor = tensor.squeeze(0).squeeze(0).data
        tokens.append(np.array(tensor))
        labels.append(word)
    
    if visualisation_model == "tsne":
        vis_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=5000, random_state=23, verbose=1, n_jobs=10)
    elif visualisation_model == "pca":
        vis_model = PCA(n_components=2)

    tokens_len = len(tokens)

    for val in res["embedding_value"]:
        tokens.append(np.array(val.data))

    new_values = vis_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values[:tokens_len]:
        x.append(value[0])
        y.append(value[1])

    x_embedding = []
    y_embedding = []
    
    for value in new_values[tokens_len:]:
        x_embedding.append(value[0])
        y_embedding.append(value[1])

    fig = plt.figure(figsize=(16, 16))
    ax = plt.axes()
    for i in range(len(x)):
        scats = plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    # what interval to use for something
    ani = animation.FuncAnimation(fig, update_plot, frames=len(x_embedding), interval=30, save_count=50, fargs=(x, y, x_embedding, y_embedding, ax, labels))
    ani.save("figs/fig-{}-{}.gif".format(res["start_word"], visualisation_model))

def save_static_graph(y, start_word, y_label):
    x = range(0, len(y)*20, 20)
    plt.plot(x, y)
    plt.xlabel("epochs")
    plt.ylabel(y_label)
    plt.title(start_word)
    plt.savefig("figs/fig-{}-{}.png".format(start_word, y_label))
    plt.clf()

class FilterVisualizer():
    def __init__(self, vocab_size, pretrained_embeddings, device, model):
        # init the network
        self.model = model.eval()
        set_trainable(self.model, False)
    
    def visualize(self, layer_name, vocab_size, words_to_idx, input_word=None, optim_steps=20):
        # create input to "train"
        d = {}
        if input_word == None:
            random_sentence = torch.LongTensor(1, 1).random_(0, vocab_size)
        else:
            try:
                random_sentence = re.split(r'(\s+)', input_word)
                random_sentence = input_word.split(" ")
                input_to_idx = []
                for word in random_sentence:
                    input_to_idx.append(words_to_idx[word])

                random_sentence = torch.LongTensor([input_to_idx])
            except:
                print("Input words not in the vocabulary")
                exit()
        
        for k, v in words_to_idx.items():
            if random_sentence[0].item() == v:
                d = {"start_word": k}
                break

        random_sentence = model.embeds_input(random_sentence)
        start_sentence = random_sentence.clone() 
 
        t = model.get_embedding()
        temp_list = []
        _, words = torch.topk(torch.mv(t.weight, random_sentence.squeeze(0).squeeze(0)), 20)
        for i, word in enumerate(words):
            for k, v in words_to_idx.items():
                if word.item() == v:
                    temp_list.append(k)
                    break

        d["closest_words_before_training"] = temp_list

        random_sentence = Variable(random_sentence, requires_grad=True)

        # hook the layer we want to hook
        for n, m in self.model.named_modules():
            if n == layer_name:
                print("Register hook for layer named {}".format(layer_name))
                m.register_forward_hook(hook_fn)
                break

        optimizer = torch.optim.Adam([random_sentence], lr=LR)

        for nm in range(optim_steps):
            if nm % 20 == 0:
                norm = torch.norm(random_sentence.squeeze(0)).item()
                try:
                    d["embedding_value"].append(random_sentence.detach().clone().squeeze(0).squeeze(0))
                except KeyError:
                    d["embedding_value"] = [random_sentence.detach().clone().squeeze(0).squeeze(0)]
                try:
                    d["norm"].append(norm)
                except KeyError:
                    d["norm"] = [norm]

            optimizer.zero_grad()
            prediction = self.model(random_sentence, embedded_input=True)
            print(prediction)

            for n, m in self.model.named_modules():
                if n == layer_name:
                    loss = m._value_hook.mean()
                    loss.backward()
                    optimizer.step()
                    if nm % 20 == 0:
                        try:
                            d["activation"].append(m._value_hook.mean())
                        except KeyError:
                            d["activation"] = [m._value_hook.mean()]
            
        # https://discuss.pytorch.org/t/vec2word-or-something-similar/2068/2
        temp_list = []
        _, words = torch.topk(torch.mv(t.weight, random_sentence.detach().clone().squeeze(0).squeeze(0)), 20)
        for i, word in enumerate(words):
            for k, v in words_to_idx.items():
                if word.item() == v:
                    temp_list.append(k)
                    break

        d["closest_words_after_training"] = temp_list

        '''
        inp = start_sentence
        prediction = self.model(inp, embedded_input=True)
        p = random_sentence.clone() / 1000
        i = 0

        while prediction.item() > 0.1:
            p = p*2
            inp = start_sentence + p
            prediction = self.model(inp, embedded_input=True)
            i += 1
            if i % 10000 == 0:
                print(i)

        start_norm = torch.norm(start_sentence.squeeze(0)).item()
        perdiction_norm = torch.norm(prediction.squeeze(0)).item()
        random_sentence_norm = torch.norm(random_sentence.squeeze(0)).item()
        print(start_norm)
        print(perdiction_norm)
        print(random_sentence_norm)
        print(p.max())
        print(p.mean())
        '''

        return d

if __name__ == "__main__":

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
    res = f.visualize("fc2", vocab_size, words_to_idx, input_word=None, optim_steps=20000)
    #save_static_graph(res["activation"], res["start_word"], "activation")
    #save_static_graph(res["norm"], res["start_word"], "norm")
    print("word {}, start_activation {}, end_activation {}".format(res["start_word"], res["activation"][0], res["activation"][-1]))
    #save_embeddings(best_model, res, words_to_idx, visualisation_model="tsne")
