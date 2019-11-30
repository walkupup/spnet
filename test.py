import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from models.spmodel import SPNet
import numpy as np

model = SPNet('p1127.txt', 128)
model.load_state_dict(torch.load('output/params_14.pth'))
model.eval()
if torch.cuda.is_available():
    model.cuda()
img = cv2.imread('0_00015.jpg', cv2.IMREAD_GRAYSCALE)
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0)
if torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))
pred = prediction.cpu().detach().numpy()
pred = pred.reshape(28, 28)
pred = np.where(pred > 0, pred, 0)
pred = np.where(pred < 1, pred, 1)
pred = pred * 255
predImage = pred.astype(np.uint8)
cv2.imshow("image", predImage)
cv2.waitKey(0)