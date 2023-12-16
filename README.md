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

위의 예시처럼 정확도 95%, 추론시간 2시간인 Teacher Model이 있다고 가정해보겠습니다. 이 잘 학습시킨 Teacher Model의 지식을 단순한 Student Model에게 전달하여 정확도 90%, 추론시간 5분의 성능을 내도록 하는 것입니다.

## Knowledge Distillaion 적용 과정
![image](https://github.com/bae60/AI/assets/146174793/f5d80690-9871-49c8-95f5-aa2707189f89)

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

#### 우선 필요한 라이브러리를 import 합니다.
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

#### Loading MNIST dataset
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



