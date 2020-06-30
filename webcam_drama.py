import numpy as np
import cv2
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToTensor
from PIL import Image, ImageOps
import random

class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        out = self(images)                      
        loss = F.binary_cross_entropy(out, targets)      
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        out = self(images)                           # Generate predictions
        loss = F.binary_cross_entropy(out, targets)  # Calculate loss
        score = F_score(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach() }
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))
        

class ProteinCnnModel2(MultilabelImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 3)
    
    def forward(self, xb):
#         return torch.sigmoid(self.network(xb))
        return F.softmax(self.network(xb), dim=1)



model = ProteinCnnModel2()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)
model.to(device)
#### for trained weights ###
model.load_state_dict(torch.load('resnet_signlang_andrew_v3.pth'))
model.eval()



labels = {
    0: 'zero',
    1: 'five',
    2: 'three'
}


def encode_label(label):
    target = torch.zeros(3)
    for l in str(label).split(' '):
        target[int(l)] = 1.
    return target

def decode_target(target, text_labels=False, threshold=0.4):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)



def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def predict_single(image):
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    preds = model(xb)
    prediction = preds[0]
    # print("Prediction: ", prediction)
    ans = decode_target(prediction, text_labels=True)
    # print('Labels:', decode_target(prediction, text_labels=True))
    return ans


cap = cv2.VideoCapture(0)

timeout = time.time() + 60*1 

game_iplist = ['zero(0)','five(1)']
game_ip = game_iplist[random.randint(0,1)]
score = 0

while(True):
    # Capture frame-by-frame
    ret, vframe = cap.read()
    frame = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
    # vframe[145:149,141:146,:] = 100
    vframe[145:149,495:499,:] = 100
    cropped_frame = frame[150:,500:,:]
#     cropped_frame = np.rollaxis(cropped_frame, 3, 1) 
    np_img = Image.fromarray(cropped_frame)
    img = np_img.resize((130,130))
    # img = ImageOps.mirror(img)
    img = ToTensor()(img)
    print('Show: ',game_ip)
    print('Score: ', score)
    ans = predict_single(img)
    if ans == game_ip:
        score = score + 1
        game_ip = game_iplist[random.randint(0,1)]

    # print(frame.shape)
    # time.sleep(5)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',vframe)
    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() > timeout:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()