---
title: "Deep Learning with Fastai - 1: Quick Start"
date: 2020-07-10T15:39:31+09:00
# draft: true

categories: ["Machine Learning"]
# subcategories: ["fastai"]
tags: ["fastai", "python", "colab"]
lightgallery: true
lightGallery: true


---
<img class="lazyload"  src="/images/fastai_1_quickstart/cover.png" />


# Introduction

`fastai`는 Pytorch 기반의 딥러닝 API로 딥러닝 기술을 누구나 쉽게 이용할 수 있도록 하는 것을 목표로 하며 [강의](https://course.fast.ai/)와 [라이브러리](https://github.com/fastai/fastai)를 제공합니다. 

이 글을 시작으로하는 ***Deep Learning with fastai*** 시리즈는 `fastai`의 강의를 정리하는 목적과 더불어, 실제로 실습을 진행하면서 새롭게 알게된 점이나 느낀점을 공유하고자 하는 목적을 가지고 있습니다.

첫강의의 내용은 **강아지와 고양이의 품종 분류**로 이미지 분류에 대해서 Quick Start 방식으로 내용을 진행하며, 이 강의 내용만으로도 `fastai` 가 추구하는 방향에 대해서 어렴풋이 알 수 있을 것입니다.

 

비단 첫강의 뿐만 아니라 강의 스타일 자체가 Top-Down 방식으로 진행되어, 우선 실습해보고 이를 기반으로 이론에 대해서 알아보는 방식이니 참고하시면 좋을 것 같습니다.

여기서 사용된 실습 자료는 Github [Fastai-Practice](https://github.com/sanghoho/Fastai-Practice) 레포지터리에서 공유되고 있으며 해당 주소는 다음과 같습니다.

- https://github.com/sanghoho/Fastai-Practice/blob/master/Week%201/Study_Week_1_fastai.ipynb

## Requirement

`fastai` 라이브러리는 `pytorch`를 기반으로 만들어졌기 때문에 GPU 연산을 지원합니다. 그렇기 때문에 실습 시 GPU를 갖춘 클라우드 서버 환경을 준비하신다면, 원활한 실습을 진행하실 수 있습니다. 이에 적합한 환경은 구글에서 제공하는 **[Colab](https://colab.research.google.com/)**으로 기본적으로 `fastai` 라이브러리가 이미 설치되어 있기 때문에, Colab 환경에서 실습을 진행하였습니다.

### Colab GPU 셋팅

우선 Colab에서 제공하는 GPU를 사용하시려면, 런타임 > 런타임 유형 변경 을 클릭하시고 노트 설정이 뜨면, 하드웨어 가속기를 GPU로 설정해주시면 됩니다. 

![colab-logo](https://colab.research.google.com/img/colab_favicon_256px.png)

![gpu-set](/images/fastai_1_quickstart/1_gpu_set.png)

### Google Drive 연동

다음은 구글 드라이브 연동 과정입니다. 

이번 실습에서는 크게 중요하지 않지만, 앞으로 구글 드라이브에서 데이터를 읽어오고 결과를 저장하는 등의 작업을 하기 위해서는 구글 드라이브와의 연동이 필수적입니다.    

![google-drive-logo](https://www.trendmicro.com/content/dam/trendmicro/global/en/partners/explore-alliance-partners/googledrive-logo.jpg)

```python
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```

위의 코드를 실행하면 구글 로그인과 연계가 되고, 최종적으로는 인증키를 반환합니다.

이 키를 복사하셔서 Colab의 결과창에 입력해주시면 해당 계정의 구글드라이브와 연동이 됩니다. 

```python
from pathlib import Path

root_dir = Path("/content/gdrive/My Drive/Colab Notebooks/" )
base_dir = root_dir / 'B-Cube-DA-1'
```

연동된 구글 드라이브의 진짜 `root_dir`은 `/content/gdrive/My Drive/` 까지 입니다. 이 경로 이후로는 자신의 구글드라이브 작업 환경에 따라 다를 것입니다. 

위의 코드는 경로 설정과 관련된 코드로, python의 `pathlib` 에 속한 `Path` 클래스를 이용하였습니다. 

초기 경로가 지정된 `Path` 클래스는 `/` 연산자를 통해 손쉽게 경로를 추가할 수 있습니다.   

### Import fastai

이번 실습에서는 이미지 데이터를 다루기 때문에, `fastai.vision` 을 import 하면 됩니다.

필요한 메소드나 속성들만 가져올 수 있지만, 편의상 `*` 를 사용하여 모두 사용가능하게 합니다.

그리고 의존하고 있는 다양한 라이브러리들이 함께 불러와지기 때문에 `pytorch`, `pandas`, `numpy`등의 함수도 별다른 import 과정 없이 사용가능합니다.     

```python
from fastai.vision import *
from fastai.metrics import error_rate
```

- 2020년 7월 15일 기준으로 fastai 라이브러리 이용시 많은 경고 메세지가 발생하는 것을 확인하실 수 있습니다.  이는 Pytorch의 버전에 따른 문제로, 다음의 코드로 Pytorch를 설치해주시면 경고 메세지 없이 사용하실 수 있습니다.

```bash
!pip install "torch==1.4" "torchvision==0.5.0"
```

## 1. `fastai` Quick Start

앞서 말씀드린 것처럼 이번 글에서는 강아지와 고양이의 품종 분류 문제를 풀기 위해 이미지 데이터를 활용합니다. 즉 이미지 분류(Image Classification)인데 `fastai`는 크게 3단계의 프로세스로 진행됩니다.

1. 데이터 준비 (`DataBunch`)
2. 모델 생성 및 학습 (`Learner`)
3. 모델 해석 (`ClassificationInterpretation`)

### 1.1. 데이터 준비

사용할 데이터는 [O. M. Parkhi et al., 2012](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)에서 사용된 [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) 데이터입니다. 이 데이터는 12종의 고양이와 25종의 개를 가지고 있고, 만들어볼 모델은 이러한 37가지의 다른 종들을 구별해 낼 것입니다. 해당 논문의 연구결과에 따르면, 2012년 기준 가장 높은 정확도는 **59.21%** 였습니다. 이는 "이미지", "머리", 그리고 "몸"을 사진으로부터 분리해서 만들어낸 다소 복잡한 모델의 성능입니다. 

`fastai` 에서는 Pytorch 공식문서에서 제공하는 예제 데이터나 기타 여러 데이터들을 쉽게 다운로드 받을 수 있도록 `untar_data` 함수와 `URLs` 클래스를 제공하고 있습니다. 이 `untar_data` 함수에 `URLs`에 속한 url을 인자로서 넘겨주면, 자동으로 데이터를 다운 받고 추출(untar)하게 됩니다. 

```python
path = untar_data(URLs.PETS)
path
```

```python
# 다운로드된 데이터 폴더의 구조를 확인
path.ls()
```

```python
path_anno = path/'annotations'
path_img = path/'images'
```

```python
# 이미지들의 파일이름을 확인하기 위한 함수 사용
fnames = get_image_files(path_img)
fnames[:5]
```

파일의 라벨링 규칙을 확인한 결과 정규표현식을 사용해야 하며, 이를 통해 `DataBunch` 생성합니다.

```python
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)
```

```python
# 샘플링된 데이터와 클래스를 확인 
data.show_batch(rows=3, figsize=(7,6))
```


{{< image src="/images/fastai_1_quickstart/2_show_batch.png" caption="" alt="show-batch" height="500px">}}

```python
# 분류 클래스를 확인하는 코드입니다.
print(data.classes)
len(data.classes),data.c
```

### 1.2. 모델 생성 및 학습

`DataBunch`를 정상적으로 생성하셨다면, `Learner`라는 것을 만들어서 모델의 생성 및 학습을 진행하실 수 있습니다.

자주 사용하게될 `Learner`는 CNN(Convolutional Neural Network) 기반으로 `cnn_learner`로 생성합니다.

- 인자 설명
    - `data`는 앞서 만든 `DataBunch` 입니다.
    - `model` 은 어떤 CNN 아키텍쳐를 쓸지에 관한 것으로, `URLs`와 유사하게 `models`라는 클래스를 사용하여 다양한 아키텍쳐 구조와 사전학습된 파라미터를 가져올 수 있습니다.
    - **`metrics`**는 학습간에 어떤 출력을 보여줄지


```python
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
```

{{< admonition type=tip title="추가설명" open=true >}}
기존 강의 자료에서는 `ConvLearner` 라고 나와있는데 `cnn_learner` 로 변경되었습니다.
{{< /admonition >}}



```python
learn.fit_one_cycle(4)
```

<!-- ![fit-one-cycle](/images/fastai_1_quickstart/3_fit_one.png) -->

{{< image src="/images/fastai_1_quickstart/3_fit_one.png" height="300px" alt="fit-one-cycle" >}}

손쉽게 약 92%의 정확도를 가지는 품종 분류 모델을 생성하였습니다!

```python
# 자신이 원하는 경로에 학습 결과를 저장할 수 있습니다.
learn.save(base_dir/ 'stage-1', return_path=True) 
```

{{< image src="/images/fastai_1_quickstart/4_learn_save.png" caption="저장된 pth 파일" alt="learn-save" >}}

### 1.3. 모델 해석

`fastai` 라이브러리는 모델을 학습하는하는 것에서 끝나는 것이 아니라, 이를 분석할 수 있는 기능 또한 포함하고 있습니다. 

학습이 완료된 `Learner` 를 `ClassificationInterpretation.from_learner()` 에 입력하여 생성합니다. 

```python
interp = ClassificationInterpretation.from_learner(learn) 

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs) # 해석에는 validatoin set 을 사용함
```

먼저 모델이 가장 혼란스러워한 데이터셋에 대해 살펴 보겠습니다. 

```python
interp.plot_top_losses(9, figsize=(15,11))
```
{{< image src="/images/fastai_1_quickstart/5_top_losses.png" caption="" alt="top_losses" height="500px" >}}


```python
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
```
{{< image src="/images/fastai_1_quickstart/6_confusion_matrix.png" caption="" alt="learn-save" height="500px">}}


Confusion Matrix를 이용하면 한눈에 어느 클래스들 간에 혼동이 잦았는지에 대해서 파악할 수 있습니다.

```python
interp.most_confused(min_val=2)
```

## 2. `fastai` Advanced Start

이번 섹션에서는 앞서 학습한 모델의 성능을 높일 수 있는 방법에 대해서 알아보고자 합니다. 

### 2.1. Unfreezing and fine-tuning

**unfreezing**은 사전학습 파라미터들을 파인튜닝(fine-tuning)하기 위해 사용됩니다. 처음에 사전 학습 파라미터를 사용하는 `Learner`를 생성하게 되면 마지막단에서 분류를 진행하는 레이어만 학습을 통해 파라미터가 조정됩니다. 즉, 이미지의 특징을 잡아주는 Convolution 레이어는 잠겨있는데, `unfreeze()` 함수로 잠금장치를 풀어주는 것과 같습니다. 

- Unfreeze를 하는 이유
    - 레딧 관련글 [[link]](https://www.reddit.com/r/learnmachinelearning/comments/a0sqg4/what_is_freezing_layers_in_fastai/)
    - fastai 포럼 [[link]](https://forums.fast.ai/t/why-do-we-need-to-unfreeze-the-learner-everytime-before-retarining-even-if-learn-fit-one-cycle-works-fine-without-learn-unfreeze/41614)

```python
learn.unfreeze()
```

```python
learn.fit_one_cycle(1)
```

### 2.2. learning rates

`fastai` 는 적합한 학습률(learning rate)를 찾아주는 `lr_find()` 함수도 제공하고 있습니다.

lr_find는 학습률을 점점 올려가면서 손실에 관한 탐색을 진행하고, 발산하게 되면 탐색을 멈추게 됩니다.   

본격적인 파인튜닝을 위해 다시 이전에 학습된 결과를 불러오겠습니다.

```python
learn.load(base_dir / 'stage-1');
```

```python
learn.unfreeze()
```

```python
# 학습률 탐색 및 시각화
learn.lr_find()
learn.recorder.plot(suggestion=True) ## 어느 부분이 적합한지 추가로 표시
```
{{< image src="/images/fastai_1_quickstart/7_lr_find.png" alt="lr-find" height= "400px">}}



```python
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
```
{{< image src="/images/fastai_1_quickstart/8_fit_one_with_unfreeze.png" alt="fit-one-unfreeze" height="200px">}}

### 2.3. Change Architecture

Deep Learning은 층이 깊어질 수록 더욱 해당 Task를 잘 학습하는 방향으로 발전하게 되었고, 가장 대표적인 아키텍쳐는 [ResNet](https://arxiv.org/pdf/1512.03385.pdf)일 것입니다. 이 아키텍쳐는 층이 깊어질수록 성능이 떨어졌던 기존의 아키텍쳐들의 문제를 해결하여 층이 깊어질수록 더 좋은 효과를 내고 있습니다. **ResNet**에 대해서는 차후의 강의에서 더욱 심도깊게 다룰 것이고, 지금은 품종분류에서 더 좋은 성능을 낼 수 있도록 34층(`resnet34`)에서 50층(`resnet50`)으로 아키텍쳐를 바꿔서 학습을 해보도록 하겠습니다. 혹시 이에 관한 읽을 자료가 필요하시다면 이미지 분류 모델의 발전 과정에 대한 [블로그 포스트](https://dnddnjs.github.io/cifar10/2018/10/09/resnet/)를 참조하시면 좋을 것 같습니다.

더욱 깊어진 아키텍쳐로 인해서 GPU 메모리가 부족하다는 경고를 보게 된다면, batch 사이즈를 줄여서 이를 해결할 수 있습니다.

```python
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2).normalize(imagenet_stats)
```

```python
learn = cnn_learner(data, models.resnet50, metrics=error_rate)
```

```python
learn.lr_find()
learn.recorder.plot()
```
{{< image src="/images/fastai_1_quickstart/9_lr_find_freeze.png" alt="lr-freeze" height="400px">}}

```python
learn.fit_one_cycle(8)
```
{{< image src="/images/fastai_1_quickstart/10_fit_resnet50.png" alt="fit-resnet50" height="400px">}}


```python
learn.save(base_dir / 'stage-1-50')
```

- 성능 향상을 위한과정

```python
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
```
{{< image src="/images/fastai_1_quickstart/11_unfreeze_resnet50.png" alt="lr-freeze" height="200px">}}

```python
learn.load(base_dir / 'stage-1-50')
```

- 결과 분석

```python
interp = ClassificationInterpretation.from_learner(learn)
```

```python
interp.most_confused(min_val=2)
```

## 3. $(+\alpha)$ MNIST Dataset

유명한 손글씨 분류 데이터셋인 MNIST 데이터셋으로 손글씨 분류기를 만들어보도록 하겠습니다. 여기서는 학습과정보다 다양한 방법으로 데이터를 준비하는 과정을 유의하여 보면 좋을 것 같습니다.

자세한 함수군들에 대한 설명은 해당 [공식문서](https://docs.fast.ai/vision.data.html#Factory-methods)를 확인해주시면 됩니다.

```python
path = untar_data(URLs.MNIST_SAMPLE); path
```

손글씨 데이터는 좌우 반전등의 데이터 변형 기법들이 적용되면 안되기 때문에, `get_transforms(do_filp=False)` 를 이용합니다.

```python
tfms = get_transforms(do_flip=False) 
```

### 3.1. `from_folder`

```python
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)
```

```python
data.show_batch(rows=3, figsize=(5,5))
```
{{< image src="/images/fastai_1_quickstart/12_mnist_batch.png" alt="lr-freeze" height="500px">}}

이번에는 error_rate가 아닌 accuracy를 표시해보도록 하였습니다.

```python
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(2)
```
{{< image src="/images/fastai_1_quickstart/13_mnist_accuracy.png" alt="lr-freeze" height="200px">}}

### 3.2. `from_csv`

```python
data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)
```

### 3.3. `from_df`

```python
df = pd.read_csv(path/'labels.csv')
df.head()
```
{{< image src="/images/fastai_1_quickstart/14_df_head.png" alt="lr-freeze" height="200px">}}

```python
data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)
data.classes
```

### 3.4. `from_name_re`

```python
fn_paths = [path/name for name in df['name']]; fn_paths[:2]
```

```python
pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)
data.classes
```

### 3.5. `from_name_func`

```python
data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,
        label_func = lambda x: '3' if '/3/' in str(x) else '7')
data.classes
```

```python
labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]
labels[:5]
```

### 3.6. `from_lists`

```python
data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24)
data.classes
```

# Conclusion

이번 글에서는 강아지와 고양이의 품종 분류 문제를 풀어나가는 fastai Quickstart를 중점적으로 다루었고, 성능 향상을 위한 몇몇 방법들, 그리고 `DataBunch`를 만들기 위한 다양한 방법들에 대해서도 알아보았습니다. 

fastai 사용해보면서 느낀점은 모델링 작업을 하다보면 편의상 필요로 하는 부분들을 잘 구현해주어 보다 쉬운 사용이 가능하다는 것입니다. 또한 뒤에서 자세하게 다루겠지만, 전이학습(Transfer Learning) 및 One Cycle Policy가 잘 적용되어 있어서 대부분의 상황에서 좋은 모델 결과를 기대할 수 있습니다.

`fastai` 시리즈는 강의를 기반으로 하지만, 포스팅 순서는 강의의 순서를 따르지는 않습니다. 앞으로의 포스팅 순서에 대해서 간략하게 말씀드리면 이번에는 이미지 분류 문제만을 다루어 보았는데, 이미지 데이터를 활용한 더욱 다양한 방법론들을 다루어 볼 것입니다. 다음으로는 테이블 데이터의 활용과 추천 시스템, 그리고 자연어 처리 (Natural Language Processing)의 순서로 진행할 예정입니다. 

감사합니다!

# Reference

1. 수업 실습 자료 [[ipynb]](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)
2. 수업 강의 영상 [[Video]](https://course.fast.ai/videos/?lesson=1) 
3. 강의 상세 설명 [[Github]](https://github.com/hiromis/notes/blob/master/Lesson1.md)
4. Pytorch 공식 튜토리얼 [[Doc]](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
