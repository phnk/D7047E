from model import create_dataloader, Net
import torch
from gensim.models import KeyedVectors
import torch.optim as optim
from save_features import SaveFeatures
from torch.autograd import Variable
import argparse

args = argparse.ArgumentParser()
args.add_argument("--load", type=str, required=False)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LR = 1e-3

def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

class FilterVisualizer():
    def __init__(self, vocab_size, pretrained_embeddings, device, model):
        # init the network
        self.model = model
        set_trainable(self.model, False)
    
    def visualize(self, layer, vocab_size, optim_steps=5):
        # create input to "train"
        random_sentence = torch.LongTensor(1, 250).random_(0, vocab_size)
        random_sentence = model.embeds_input(random_sentence)
        random_sentence = Variable(random_sentence, requires_grad=True)

        # hook to a layer. currently cant hook to LSTM because it gives an error
        activations = SaveFeatures(list(self.model.children())[layer])
        

        input_optim = torch.optim.Adam([random_sentence], lr=LR)

        for _ in range(optim_steps):
            input_optim.zero_grad()
            self.model(random_sentence, embedded_input=True)
            loss = -activations.features.mean() #get mean, no filter used because of NLP tasks
            loss.backward()
            input_optim.step()

        print(input_optim)

if __name__ == "__main__":
    print("device to use: {}".format(DEVICE))
 
    # load pretrained model from file
    pretrained_embeddings = KeyedVectors.load("models/word2vec_m5_s300_w8_s0_h0_n5_i10")

    train_loader, val_loader, test_loader, vocab_size = create_dataloader("data/train.csv")

    model = Net(vocab_size=vocab_size, pretrained_embeddings=pretrained_embeddings, device=DEVICE)
    
    # train the network
    if args["load"] is None:
        best_model = model._train(train_loader, val_loader, test_loader, save=True)
    else:
        best_model = model.load(args["load"])

        
    f = FilterVisualizer(vocab_size=vocab_size, pretrained_embeddings=pretrained_embeddings, device=DEVICE, model=best_model)
    f.visualize(2, vocab_size)
   
