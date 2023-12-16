# Knowledge-Distillation
AI2 기말 프로젝트_정보융합전공 2020017210 배유경

NIPS 2014 workshop에서 발표한 "Distilling the Knowledge in a Neural Network"논문과 논문구현 코드에 대해 소개하겠습니다.

## Knowledge Distillation의 등장배경
딥러닝의 발전으로 모델은 점점 더 깊어지고 크기가 커지고 있습니다. 이 과정에서 여러 문제점이 발생합니다.

**1) 모델배포 문제**

**2) 비용, 추론시간 증가 문제**

**3) 실용적 측면에서의 문제**

복잡한 구조를 가진 모델의 성능을 향상시키기 위해 여러모델의 결과를 합쳐서 활용하는 앙상블 기법이 대표적으로 활용되고 있습니다. 이를 통해 아무리 복잡한 모델일지라도 최고의 정확도를 내도록 설계되고 있습니다. 하지만 이 모델은 실제 사용자에게 적합하지 않을 수 있습니다. 모바일이나 자동차와 같이 GPU를 사용할 수 없는 환경에서는 이런 복잡한 딥러닝 모델을 사용하기 어렵습니다. 또 앙상블 기법을 통해 모델의 성능은 높아질 수 있겠지만, 대규모 모델은 여러 모델이 합쳐진 만큼 비용이 과도하게 들어가고, 같은 자원이라면 추론시간이 증가 할 수 밖에 없습니다. 그렇기때문에 실시간으로 수행되어야하는 작업에 있어서는 적용되기가 힘듭니다. 이에따라 **딥러닝 알고리즘을 더 빠르고 가볍게 할 필요성을 가지게 되었습니다.** 이때 등장한 개념이 Knowledge Distillation 입니다. 

## Knowledge Distillation 이란?
한국어로 **'지식 증류'** 라고 하는데요. 증류는 불순물이 섞여있는 혼합물에 온도를 가하여 원하는 성분을 분리시키는 방법입니다. 이것을 딥러닝에 적용시키면 **"불필요하게 많은 파라미터가 사용되는 기존의 복잡한 모델들로부터, 보다 단순화된 모델에 지식을 전달해서 핵심 부분은 살리고, 불필요한 부분은 제거하여 모델속도를 개선하는 것"** 입니다.

Knowledge Distillation 에는 Teacher Model과 Student Model이 필요합니다.
> Teacher Model (선생모델) : 높은 예측 정확도를 가진 복잡한 모델

 
> Student Model (학생모델) : 선생모델의 지식을 받는 얕고 단순한 모델

선생님이 학생에게 가르침을 통해 지식을 주는 것 처럼 **잘 학습된 Teacher Model의 지식을 Student Model에게 전달하여 비슷한 성능을 내고자 하는 것이 Knowledge Distillaion의 목적입니다.**

![스크린샷 2023-12-13 001529](https://github.com/bae60/AI/assets/146174793/2d23749e-8045-4277-82b6-04f278b26baf) 
<p align="center">
[그림1]
</p>


위의 그림처럼 정확도 95%, 추론시간 2시간인 Teacher Model이 있다고 가정해보겠습니다. 이 잘 학습시킨 Teacher Model의 지식을 단순한 Student Model에게 전달하여 정확도 90%, 추론시간 5분의 성능을 내도록 하는 것입니다.

## Knowledge Distillaion 적용 과정
![image](https://github.com/bae60/AI/assets/146174793/f5d80690-9871-49c8-95f5-aa2707189f89)
<p align="center">
[그림2]
</p>

**1) Soft Target**

![2제목 없음](https://github.com/bae60/AI/assets/146174793/4d47b99f-3873-4455-8fe9-a1c3ee8818f4)

일반적인 분류모델에서는 입력값이 네트워크를 통과하면 마지막에 로짓값이 산출되게 됩니다. 여기에 softmax를 취하여 합이 1인 확률값으로 변형합니다. 가장 높은 확률값을 보이는 클래스에는 1을 부여하고 나머지의 클래스에 대해서는 0을 부여하여 아웃풋을 도출하는 One-hot 인코딩 방식을 사용합니다. 본 논문에서는 **1과 0으로만 이루어진 예측값을 Hard Target** 이라고 이름 붙혔습니다. Knowledge Distillaion은 Hard Target이 아닌 Soft Target을 사용하고 있는데요.

![3제목 없음](https://github.com/bae60/AI/assets/146174793/170d806e-fdd1-4604-b87b-e6760e0ef40c)

**Soft Target 은 One-hot 인코딩을 거치지 않은 예측결과의 확률분포를 의미합니다.** 위의 Soft Target을 보면 입력이미지가 토끼보다는 고양이에 가깝고 고양이보다는 강아지에 더 가깝다는 것, 고양이와 강아지가 어느정도 비슷한 특징이 있다는 것을 알 수 있습니다. 이런 점에서 봤을 때, 기존의 정답데이터인 Hard Target보다 Soft Target이 갖는 정보가 더 크므로 이 지식을 Student Model에게 전달하는 것입니다.

이렇게 Soft Target을 통해 정보손실을 최소화했지만 가장 큰 로짓값을 갖는 노드의 출력값은 1과 매우 가까운 값을 가지고, 나머지는 0에 가까운 값으로 매핑되는 문제점을 가지고 있습니다. 이 문제점을 개선하기 위해 **𝝉(temperature)** 라는 하이퍼파라미터를 Softmax 함수에 추가하여 약간의 변형을 취해줍니다. 𝝉는 일종의 Scaling 역할을 합니다. 
![4제목 없음](https://github.com/bae60/AI/assets/146174793/aa94c93f-155a-4155-81ac-e5d59b70c918)

**2) 손실함수**

![image](https://github.com/bae60/AI/assets/146174793/d8f12b77-9752-48c1-aed7-25f68165dd10)
𝑆 = Student model , 𝑇 = Teacher model 


앞에 수식은 Distillaion loss를 구합니다. S가 구한 soft prediction, T가 구한 soft label의 차이로 손실을 구합니다. 
뒤에 수식은 일반적인 신경망 학습에 사용하는 손실입니다. S가 구한 soft prediction과 원래 데이터가 가지고 있는 hard label의 차이로 손실을 구합니다. 
이 두가지 손실의 합을 최종 손실로 삼아 학습합니다.

## Knowledge Distillation 논문구현
본 코드는 https://github.com/Seonghoon-Yu/AI_Paper_Review/blob/master/Classification/Knowledge_distillation(2014).ipynb 의 코드를 참고하였습니다.

### 우선 필요한 라이브러리를 import 합니다.
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import time
import os
import copy
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
%matplotlib inline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Loading MNIST dataset
MNINST dataset을 불러오는 과정입니다.
```
# make directorch to save dataset
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSerror:
        print('Error')
createFolder('./data')
```
```
# define transformation
ds_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,),(0.3081,))
])
```

dataset을 생성합니다.
```
# load MNIST dataset
train_ds = datasets.MNIST('/content/data',train=True, download=True, transform=ds_transform)
val_ds = datasets.MNIST('/content/data',train=False, download=True, transform=ds_transform)
```

데이더 로더를 생성합니다.
```
# define data loader
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size = 128, shuffle=True)
```

샘플 이미지를 확인합니다.
```
# check sample image
for x, y in train_dl:
    print(x.shape, y.shape)
    break

num = 4
img = x[:num]

plt.figure(figsize=(15,15))
for i in range(num):
    plt.subplot(1,num+1,i+1)
    plt.imshow(to_pil_image(0.1307*img[i]+0.3081), cmap='gray')
```
torch.Size([64, 1, 28, 28]) torch.Size([64])
![download](https://github.com/HY-AI2-Projects/Knowledge-Distillation/assets/146174793/27c5925f-2c79-4b0f-b72c-76ec61b121ae)

### Define Teacher model
Knowledge distillation을 하기 위해서 soft label을 얻기 위한 teacher model을 먼저 학습해야 합니다. 따라서 teacher model을 정의합니다.

```
class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 1200)
        self.bn1 = nn.BatchNorm1d(1200)
        self.fc2 = nn.Linear(1200,1200)
        self.bn2 = nn.BatchNorm1d(1200)
        self.fc3 = nn.Linear(1200, 10)
    
    def forward(self,x):
        x = x.view(-1, 28*28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x,p=0.8)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x,p=0.8)
        x = self.fc3(x)
        return x
```

```
# check
x = torch.randn(16,1,28,28).to(device)
teacher = Teacher().to(device)
output = teacher(x)
print(output.shape)
```
<p align="center">
torch.Size([16, 10])
</p>



가중치를 초기화합니다.
```
# weight initialization
def initialize_weights(model):
    classname = model.__class__.__name__
    # fc layer
    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    # batchnorm
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

teacher.apply(initialize_weights);
```

### Train teacher model
teacher model을 학습합니다.
```
# loss function
loss_func = nn.CrossEntropyLoss()

# optimizer
opt = optim.Adam(teacher.parameters())

# lr scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)
```

```
# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    
    return loss_b.item(), metric_b


# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data
    return loss, metric


# function to start training
def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    opt=params['optimizer']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    sanity_check=params['sanity_check']
    lr_scheduler=params['lr_scheduler']
    path2weights=params['path2weights']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr= {}'.format(epoch, num_epochs-1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print('Copied best model weights!')

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history
```

```
# set hyper parameters
params_train = {
    'num_epochs':30,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_dl,
    'val_dl':val_dl,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/teacher_weights.pt',
}

createFolder('./models')
```

30 epoch 학습합니다.
```
teacher, loss_hist, metric_hist = train_val(teacher, params_train)
```
![스크린샷 2023-12-16 193447](https://github.com/HY-AI2-Projects/Knowledge-Distillation/assets/146174793/9e867c25-a9fb-4b55-aa44-a36899e815d8)
#### teacher model의 accuracy 96.3 나왔습니다.


loss와 accuracy를 시각화합니다.
```
num_epochs = params_train['num_epochs']

# Plot train-val loss
plt.title('Train-Val Loss')
plt.plot(range(1, num_epochs+1), loss_hist['train'], label='train')
plt.plot(range(1, num_epochs+1), loss_hist['val'], label='val')
plt.ylabel('Loss')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()

# plot train-val accuracy
plt.title('Train-Val Accuracy')
plt.plot(range(1, num_epochs+1), metric_hist['train'], label='train')
plt.plot(range(1, num_epochs+1), metric_hist['val'], label='val')
plt.ylabel('Accuracy')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()
```
![download](https://github.com/HY-AI2-Projects/Knowledge-Distillation/assets/146174793/84c62a86-9826-422e-9fd8-b40bc2c65c5b)
![download](https://github.com/HY-AI2-Projects/Knowledge-Distillation/assets/146174793/7d110750-68f2-44a3-b19f-13bd6ac7a52d)

### Define Student model
이제 teacher의 지식을 transfer할 student model을 정의합니다.
```
class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 800)
        self.bn1 = nn.BatchNorm1d(800)
        self.fc2 = nn.Linear(800,800)
        self.bn2 = nn.BatchNorm1d(800)
        self.fc3 = nn.Linear(800,10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
```

```
# check
x = torch.randn(16,1,28,28).to(device)
student = Student().to(device)
output = student(x)
print(output.shape)
```
<p align="center">
torch.Size([16, 10])
</p>


가중치를 초기화합니다.
```
# weight initialization
def initialize_weights(model):
    classname = model.__class__.__name__
    # fc layer
    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    # batchnorm
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

student.apply(initialize_weights);
```

### Knowledge distillation
이제 앞서 설명드린 teacher model의 soft label을 사용하여 student model을 knowledge distillation loss로 학습합니다.
```
teacher = Teacher().to(device)
# load weight
teacher.load_state_dict(torch.load('/content/models/teacher_weights.pt'))

student = Student().to(device)

# optimizer
opt = optim.Adam(student.parameters())
```

```
# knowledge distillation loss
def distillation(y, labels, teacher_scores, T, alpha):
    # distillation loss + classification loss
    # y: student
    # labels: hard label
    # teacher_scores: soft label
    return nn.KLDivLoss()(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2.0 + alpha) + F.cross_entropy(y,labels) * (1.-alpha)

# val loss
loss_func = nn.CrossEntropyLoss()
```

```
def distill_loss_batch(output, target, teacher_output, loss_fn=distillation, opt=opt):
    loss_b = loss_fn(output, target, teacher_output, T=20.0, alpha=0.7)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b
```

100epoch 학습하겠습니다.
```
num_epochs= 100

loss_history = {'train': [], 'val': []}
metric_history = {'train': [], 'val': []}

best_loss = float('inf')
start_time = time.time()

for epoch in range(num_epochs):
    current_lr = get_lr(opt)
    print('Epoch {}/{}, current lr= {}'.format(epoch, num_epochs-1, current_lr))

    # train
    student.train()

    running_loss = 0.0
    running_metric = 0.0
    len_data = len(train_dl.dataset)

    for xb, yb in train_dl:
        xb = xb.to(device)
        yb = yb.to(device)

        output = student(xb)
        teacher_output = teacher(xb).detach()
        loss_b, metric_b = distill_loss_batch(output, yb, teacher_output, loss_fn=distillation, opt=opt)
        running_loss += loss_b
        running_metric_b = metric_b
    train_loss = running_loss / len_data
    train_metric = running_metric / len_data

    loss_history['train'].append(train_loss)
    metric_history['train'].append(train_metric)

    # validation
    student.eval()
    with torch.no_grad():
        val_loss, val_metric = loss_epoch(student, loss_func, val_dl)
    loss_history['val'].append(val_loss)
    metric_history['val'].append(val_metric)


    lr_scheduler.step(val_loss)

    print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
    print('-'*10)
```
![스크린샷 2023-12-16 200619](https://github.com/HY-AI2-Projects/Knowledge-Distillation/assets/146174793/37505b5d-fd16-41d8-a490-928adc691186)

#### teacher model보다 accuracy가 1.88% 향상되었습니다 !

loss와 accuracy를 시각화합니다.
```
# Plot train-val loss
plt.title('Train-Val Loss')
plt.plot(range(1, num_epochs+1), loss_hist['train'], label='train')
plt.plot(range(1, num_epochs+1), loss_hist['val'], label='val')
plt.ylabel('Loss')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()

# plot train-val accuracy
plt.title('Train-Val Accuracy')
plt.plot(range(1, num_epochs+1), metric_hist['train'], label='train')
plt.plot(range(1, num_epochs+1), metric_hist['val'], label='val')
plt.ylabel('Accuracy')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()
```
![download](https://github.com/HY-AI2-Projects/Knowledge-Distillation/assets/146174793/c12d6fe6-0d2d-4a77-848a-e871e456c1e4)
![download](https://github.com/HY-AI2-Projects/Knowledge-Distillation/assets/146174793/c5ea590a-bb97-4716-bb5b-fab9cd31b2b8)

## 결론
* 큰 모델의 지식을 작은 네트워크에 전달 할 수 있는 방법이 Knowledge distillation입니다.
* Knowledge distillation 구현코드를 실행해보며 실제 딥러닝 서비스에서 유용하게 활용될 수 있음을 알았습니다.

## 참고자료
* [Knowledge distillation 논문](https://ffighting.net/deep-learning-paper-review/deep-learning-paper-guide/deep-learning-paper-guide/#Knowledge_Distillation)
* [그림1](https://velog.io/@dldydldy75)
* [그림2](https://intellabs.github.io/distiller/knowledge_distillation.html)
