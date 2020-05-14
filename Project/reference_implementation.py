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
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
import matplotlib.animation as animation

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, required=False)
args = parser.parse_args()
writer = SummaryWriter('runs/')


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LR = 1e-3

def hook_fn(module, input, output):
    setattr(module, "_value_hook", output)

def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def update_plot(i, x_embedding, y_embedding, fig):
    p = (x_embedding, y_embedding)
    
    return scat,

def save_embeddings(model, res, visualisation_model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    i = 0
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
        i += 1
        if i == 20:
            break

    #writer.add_embedding(np.array(tokens), metadata=np.array(labels))
    #writer.close()
    # MAKE THIS WORK FOR PCA
    # TRY TO FIGURE OUT HOW TO UPDATE THE WORD THAT WE ARE TRAINING INSIDE A VISUALISATION TOOL SO WE CAN SEE HOW IT MOVES
    # MAYBE THATS DECENT RESULTS FOR THE PROJECT?

    if visualisation_model == "tsne":
        vis_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=5000, random_state=23, verbose=1, n_jobs=10)
    elif visualisation_model == "pca":
        vis_model = PCA(n_components=2)
 
    new_values = vis_model.fit_transform(tokens)
    new_embedding_values = vis_model.fit_transform(res[0]["embedding_value"])

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    x_embedding = []
    y_embedding = []
    for value in new_embedding_values:
        x_embedding.append(value[0])
        y_embedding.append(value[1])

    fig = plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        scat = plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    # animate one scatter

    ani = animation.FuncAnimation(fig, update_plot, frames=100, interval=50, fargs=(x_embedding, y_embedding, fig))
    ani.save("fig.gif")

class FilterVisualizer():
    def __init__(self, vocab_size, pretrained_embeddings, device, model):
        # init the network
        self.model = model.eval()
        set_trainable(self.model, False)
    
    def visualize(self, layer_name, vocab_size, words_to_idx, optim_steps=20):
        # create input to "train"
        d = []
        random_sentence = torch.LongTensor(1, 1).random_(0, vocab_size)
        for j in range(random_sentence.size()[1]):
            for k, v in words_to_idx.items():
                if random_sentence[0][j].item() == v:
                    d.append({"start_word": k})
                    break

        random_sentence = model.embeds_input(random_sentence)
        start_sentence = random_sentence.clone() 

        t = model.get_embedding()
        # https://discuss.pytorch.org/t/vec2word-or-something-similar/2068/2
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        normalized_embedding = t.weight/((t.weight**2).sum(0)**0.5).expand_as(t.weight)
        for j in range(random_sentence.size()[1]):
            temp_list = []
            _, words = torch.topk(cos(normalized_embedding, random_sentence.squeeze(0)[j]), 5)
            for i, word in enumerate(words):
                for k, v in words_to_idx.items():
                    if word.item() == v:
                        temp_list.append(k)
                        break

            d[j]["cloest_words_before_training"] = temp_list

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
            prediction = self.model(random_sentence, embedded_input=True)
            for n, m in self.model.named_modules():
                if n == layer_name:
                    loss = -m._value_hook.mean()
                    loss.backward()
                    optimizer.step()

            if nm % 100 == 0:
                for j in range(random_sentence.size()[1]):
                    c_s = cos(random_sentence.squeeze(0)[j], start_sentence.squeeze(0)[j]).item()
                    norm = torch.norm(random_sentence.squeeze(0)[j]).item()
                    
                    
                    try:
                        d[j]["embedding_value"].append(np.array(random_sentence.squeeze(0)[j].data))
                    except KeyError:
                        d[j]["embedding_value"] = [np.array(random_sentence.squeeze(0)[j].data)]
                    
                    try:
                        d[j]["cos_sim"].append(c_s)
                    except KeyError:
                        d[j]["cos_sim"] = [c_s]

                    try:
                        d[j]["norm"].append(norm)
                    except KeyError:
                        d[j]["norm"] = [norm]
            
        # https://discuss.pytorch.org/t/vec2word-or-something-similar/2068/2
        normalized_embedding = t.weight/((t.weight**2).sum(0)**0.5).expand_as(t.weight)
        for j in range(random_sentence.size()[1]):
            temp_list = []
            _, words = torch.topk(cos(normalized_embedding, random_sentence.squeeze(0)[j]), 5)
            for i, word in enumerate(words):
                for k, v in words_to_idx.items():
                    if word.item() == v:
                        temp_list.append(k)
                        break

            d[j]["closest_word_after_training"] = temp_list

        return d

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
    res = f.visualize("fc1", vocab_size, words_to_idx, optim_steps=2000)
    save_embeddings(pretrained_embeddings, res, visualisation_model="pca")
