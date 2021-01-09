---
title: "Machine Learning from Scratch - 1: Introduction"
date: 2021-01-09T02:29:14+09:00
draft: true

resources:
- name: featured-image
  src: 
- name: featured-image-preview
  src: 

tags: ["python", "linear algebra", "gradient descent"]
categories: ["Machine Learning"]

description:

lightgallery: true

---

<img class="lazyload" src="/images/ml_from_scratch/ml_cover.png" />

# 시리즈를 시작하면서

머신러닝의 개념을 처음 접한 것은 대학에 입학했던 2016년도 였고, 구글 [DeepMind](https://deepmind.com/) 사의 AlphaGo와 이세돌의 대국이 이루어지고 있었습니다. 과거의 데이터에서 암묵적인 패턴과 규칙을 학습하고, 새로운 데이터에 이를 적용하여 즉각적으로 충분한 대답을 추론한다는 아이디어는 명시적인 지식만을 습득해왔던 저에게 있어서 정말 놀라웠습니다. 그렇기 때문에 어떻게 하면 과거의 데이터에서 패턴을 학습하여 모델로 만들 수 있는지에 대해서 강의, 책, 스터디, 블로그 포스트 등을 전전하며 조금씩 공부해왔으나, 정돈된 자료의 형태로 기록되어 있지 않아서 항상 아쉬움이 남고는 했습니다. 

이 일련의 ***Machine Learning from Scratch*** 시리즈는 앞선 개인적인 고민을 해소하고, 이미 좋은 자료가 많이 공유되어 있지만 조금이라도 머신러닝을 새롭게 혹은 계속해서 배우고 있으신 분들에게 도움이 되었으면 좋겠다는 마음으로 연재를 시작하게 되었습니다. 연재의 전체적인 순서는 머신 러닝을 배우는데 있어서 가장 기초가 된다고 할 수 있는 [Coursera](https://www.coursera.org/) 앤드류 응 교수님의 강의, [Machine Learning](https://www.coursera.org/learn/machine-learning)을 기반으로 할 것이며, 대학에서 강의를 들었던 이재식 교수님의 저서 [데이터 애널리틱스](https://wikibook.co.kr/data-analytics/)(2020, 위키북스)의 내용도 함께 정리하고자 합니다. 

 저도 공부하는 계속해서 공부하는 입장이기에 제가 아는 선에서는 최대한 수리적인 내용(e.g., 선형대수, 통계 등등)을 담을 수 있고자 하였고, 이부분이 아니더라도 Python `numpy`를 이용한 구현이나 모델의 인사이트를 포함한 경영학적 함의를 기술하여 이 글을 읽고 계신 여러 분야의 독자 분들께 도움이 되었으면 좋겠습니다.

P.S. 밑바닥 부터 시작하는(from Scratch)라는 명칭은 이미 다들 아시겠지만, 제가 개인적으로 좋아하는 시리즈인  O'reilly 사의 [밑바닥 부터 시작하는 데이터과학(Data Science from Scratch)](https://blog.insightbook.co.kr/2016/05/27/%eb%8d%b0%ec%9d%b4%ed%84%b0-%ea%b3%bc%ed%95%99%ec%97%90-%ed%95%84%ec%9a%94%ed%95%9c-%ea%b8%b0%ec%b4%88-%ec%9d%b4%eb%a1%a0%ea%b3%bc-%ed%94%84%eb%a1%9c%ea%b7%b8%eb%9e%98%eb%b0%8d-%eb%91%90-%eb%a7%88/), [밑바닥 부터 시작하는 딥러닝(Deep Learning from Scratch)](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)를 패러디한 것 입니다.


---

## 1. Machine Learning with `numpy`

> `numpy`는 수학 및 과학 연산을 위한 Python의 기본 패키지입니다. [[공식 홈페이지]](https://numpy.org/)

코세라 Machine Learning 수업의 경우, 수치계산용 언어인 [MATLAB](https://www.mathworks.com/products/matlab.html) 혹은 [Octave](https://www.gnu.org/software/octave/index)로 과제를 진행하게 됩니다. **벡터** 및 **행렬** 연산이 언어 자체에 정의되어 있으므로 별다른 라이브러리를 사용할 필요 없이 모델의 구현이 가능하다는 장점이 있습니다. 하지만 그 본질은 결국 벡터와 행렬 연산이기 때문에, 어떤 프로그래밍 언어로 구현하시더라도 원리만 이해한다면 쉽게 구현하실 수 있을 것입니다. 

본래 수강할때는 자료에 맞추어 Octave로 구현했었으나, 이 시리즈에서는 Python으로 구현하고자 합니다. 그 중 `numpy` 라는 라이브러리를 사용할 것인데, `C`/`C++`, `Fortran`으로 작성되어 빠르면서도 정교한 수학 연산, 특히 **벡터** 및 **행렬**(선형대수) 연산이 가능하기 때문입니다.      

- `numpy`의 강점
    - a powerful N-dimensional array object
    - sophisticated (broadcasting) functions
    - tools for integrating C/C++ and Fortran code
    - useful linear algebra, Fourier transform, and random number capabilities

또한 `numpy`의 array 자료형은 Python의 딥러닝 라이브러리 양대산맥이라 할 수 있는 [Pytorch](https://pytorch.org/)의 tensor 자료형과도 쉽게 상호 전환이 되기 때문에 실질적인 모델링에도 부분적으로 활용될 수 있습니다. 추가로 `numpy` 기반의 연산과 모델링을 그대로 GPU 프로그래밍으로 전환시킬 수 있는 CUDA 기반의 [CuPy](https://cupy.dev/)라는 라이브러리도 존재하고 있습니다. 따라서 본인이 밑바닥부터 쌓아올리며 공부한 코드를 몇줄의 수정만으로도 실제 프로젝트에 적용해볼 수 있을 것입니다. 

--- 

## 2. Linear Algebra with `numpy`

### 2.1. `numpy` 기초

관습적으로 `numpy`를 import 할때, `np`라는 약어를 사용하고 있습니다.

```python
import numpy as np
```

#### 배열 생성

`numpy`의 자료 클래스를 **ndarray**(N-Dimensional Array)라고 합니다.

- `np.array()`
    - 기존의 python `list`, `tuple` 등으로 생성
    - `dtype`을 조정하여 데이터 타입 지정 가능

    ```python
    print(np.array([2,3,4]))                         # 리스트를 이용한 array 생성
    print(np.array([(1.5,2,3), (4,5,6)]))            # 튜플을 이용한 array 생성

    a = np.array( [ [1,2], [3,4] ], dtype=complex)   # 데이터타입 지정 (복소수)
    print(type(a))
    ```

- `np.arrange()`
    - 범위를 지정하여 array 생성
    - 첫번째와 두번째 인자로 range를 결정하고, 세번째 인자로 간격을 결정

    ```python
    np.arange(10, 30, 5) 
    ## Output: array([10, 15, 20, 25])  
    ```

- `np.zeros()` or `np.ones()` or `np.empty()`
    - 지정된 사이즈만큼의 array 생성 (순서대로 각각 0, 1, 초기화되지 않은 값)
    ```python
    np.zeros((3,4)) # 3행 4열 크기의 모든 값이 0인 2차원 배열

    # Out
    # array([[ 0.,  0.,  0.,  0.],
    #        [ 0.,  0.,  0.,  0.],
    #        [ 0.,  0.,  0.,  0.]])
    ```


    ---

    ```python
    np.ones((2,3,4), dtype=np.int16) # (2, 3, 4) 크기의 모든 값이 정수 1인 3차원 배열

    # Out
    # array([[[1, 1, 1, 1],
    #         [1, 1, 1, 1],
    #         [1, 1, 1, 1]],
    #        [[1, 1, 1, 1],
    #         [1, 1, 1, 1],
    #         [1, 1, 1, 1]]], dtype=int16)
    ```
    ---

    ```python
    np.empty((2,3), dtype=np.double) # (2, 3) 크기의 초기화되지 않은 값을 실수로 가지는 2차원 배열 

    # Out
    # array([[ 0.,  0.,  0.],
    #       [ 0.,  0.,  0.]])

    ```


#### 산술 연산

`ndarray`는 기초적인 산술 연산을 지원하는데, **`+`,`-`, `*`, `/`**를 지원합니다. 이 연산들은 같은 자리의 성분끼리 연산되는 *element-wise* 방식으로, 기본적으로는 원소의 수가 같을 때 연산이 가능합니다. 

```python
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y) # 원소별 덧셈      [1.0 + 2.0, 2.0 + 4.0, 3.0 + 6.0]
print(x - y) # 원소별 뺄셈      [1.0 - 2.0, 2.0 - 4.0, 3.0 - 6.0]
print(x * y) # 원소별 곱셈      [1.0 * 2.0, 2.0 * 4.0, 3.0 * 6.0]
print(x / y) # 원소별 나눗셈    [1.0 / 2.0, 2.0 / 4.0, 3.0 / 6.0]
```

다만 다음 *그림1*과 같이 **boradcasting** 이라는 방법으로 서로 다른 크기의 `ndarray` 간에도 연산이 가능한 경우가 있습니다. 따라서 본인의 의도에만 맞게 사용된다면 배열의 shape를 수정할 필요없이 바로 산술 연산을 적용할 수 있다는 장점이 있습니다.



{{< image src="/images/ml_from_scratch/1/Untitled.png" caption="그림1. broadcasting의 예시" alt="" height="1000px">}}

[출처: "[Computation on Arrays: Broadcasting](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html)"]

```python
x = np.arange(3)
x + 5 # [0 + 5, 1 + 5, 2 + 5]
```

#### N 차원 배열

*그림2* 에서는 머신러닝을 적용하는 데이터가 이루고 있는 대부분의 형태를 보여주고 있습니다. 여기서는 소개하지 않겠지만, Pandas의 Series 형태의 데이터가 1D array와 같은 모양을 보이게 될 것이고, 데이터 프레임의 경우 2D array의 형태를 보여주게 될 것입니다. 3D array의 경우 주로 sequence를 가진 데이터를 다루게될 때 자주 접할 수 있는 차원이며, 4D array는 이미지 데이터를 다루게된다면 접하게되실 수도 있을 것 같습니다. 앞으로 연재하게될 데이터의 자료형태에 따른 모델링에서 해당 부분을 더욱 자세하게 다룰 수 있도록 하겠고, 여기서는 각 차원을 어떻게 생성하는지만 보여드리도록 하겠습니다.

{{< image src="/images/ml_from_scratch/1/Untitled 1.png" caption="그림2. 1 ~ 3차원의 배열에 대한 시각적 자료와 축(axis)의 위치" alt="" height="1000px">}}

[출처: "[파이썬 데이터 사이언스 Cheat Sheet: NumPy 기초, 기본](http://taewan.kim/post/numpy_cheat_sheet/)"]

```python
A = np.array([1, 2, 3])               # 1D array
B = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array
C = np.array([[[1, 2, 3], [4, 5, 6]], # 3D array 
              [[7, 8, 9], [10, 11, 12]]])
print(A.shape, B.shape, C.shape) # (3,) (2, 3) (2, 2, 3)
```

### 2.2. 벡터 및 행렬 연산

#### 벡터 연산

다음과 같은 벡터의 연산을 어떻게 구현할 수 있을지에 대해서도 한번 고민해보시면 좋을 것 같습니다.

$$\vec{a} = (1, 2, 3)$$

$$\vec{b} = (4, 5, 6)$$

$$\therefore \vec{a} + \vec{b} = (5, 7, 9)$$

`numpy`를 활용하면 위에서 봤던 것처럼 아주 간단하게 벡터를 생성하고 연산이 가능한데, 우선 기본적인 python만을 통해서 구현한 결과는 다음과 같습니다.

- python `list`
    ```python
    a = [1, 2, 3]
    b = [4, 5, 6]

    print(a + b)                                 # python의 리스트에서는 이게 아니다.
    print([a_i + b_i for a_i, b_i in zip(a, b)]) # zip 함수를 써서 묶고, list comprehension

    def vector_sum(a, b):                        # 함수화
        return [a_i + b_i for a_i, b_i in zip(a, b)]
    ```

- `numpy` **ndarray**
    ```python
    import numpy as np

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    print(a + b)
    ```

#### 행렬 연산

기존의 ndarray 형태로도 물론 행렬에 관한 연산을 수행할 수 있지만, numpy에서는 보다 간편하게 행렬의 연산을 가능하게 하는 matrix의 데이터형도 제공하고 있습니다. np.array로 생성하던 것처럼 np.mat으로 생성할 수 있으며, 이미 존재하는 ndarray를 입력으로 받아서도 생성할 수 있습니다.

```python
# numpy.ndarray
A = np.array([[1, 2], [2, 3]])
B = np.array([[1, 0], [2, 5]])

# numpy.matrix
C = np.mat(A) 
D = np.mat([[1, 0], [2, 5]])
```

두 데이터형의 연산을 비교하면 꽤나 유의미한 차이를 발견할 수 있습니다. 바로 기본 곱셈 연산을 어떻게 처리하는지에 관한 것인데, ndarray의 경우 같은 위치에 있는 원소간에 곱을 하는 element-wise 곱셈을 하지만 matrix의 경우 행렬곱 연산을 하게 됩니다. ndarray에서는 행렬곱을 하기 위해 np.dot(A, B)를 해야하므로 조금 더 직관적으로 행렬 연산을 통한 머신러닝 모델 구축을 하기 위해서는 matrix 데이터형이 조금 더 적합한 것을 알 수 있습니다. 

```python
print(A * B) # elemnet-wise mul
print(np.dot(A, B)) # or A.dot(B)
print(C * D) # 행렬의 곱

```
```
# Out 1
[[ 1  0]
[ 4 15]]

# Out 2
[[ 5 10]
[ 8 15]]

# Out 3
[[ 5 10]
[ 8 15]]
```

행렬곱이 아닌 행렬 간 덧셈, 그리고 스칼라곱은 두 데이터형에서 모두 예상하는 것처럼 작동합니다.

```python
print(A + B)
print(C + D)

print(3 * A)
print(3 * C)
```

```
# Out 행렬간 덧셈
[[2 2]
 [4 8]]

# Out 스칼라곱
[[3 6]
 [6 9]]
```

다음으로는 특수한 연산인 **전치(transpose)** 행렬, **역(inverse)** 행렬, 그리고 **특이값 분해(singular value decomposition)**에 대해서 알아보도록 하겠습니다.

- 전치(transpose) 행렬
    - 전치 행렬은 행렬의 행과 열을 바꾸는 것으로 `numpy`의 배열의 property 중 `T`를 사용하여 쉽게 구할 수 있습니다.
    - 향후 예측 혹은 분류 모델을 만들게 될 때, **여러 feature 열들을 포함한 행렬**과 **가중치 행렬** 등을 곱하기 위해 중요하게 활용될 것입니다.
    ```python
        B.T
    # Out
    # array([[1, 4],
    #        [2, 5],
    #        [3, 6]])
    ```

- 역(inverse) 행렬
    - **ndarray**의 경우 `numpy.linalg.inv()` 라는 함수를 통해 구할 수 있습니다.
    - **matrix**의 경우 간단하게 속성 중 `I`라는 속성을 호출하면 됩니다.
    - ⚠ 다만 주의할 것은 **ndarray**의 경우 shape가 정방 행렬이 아니라면 pseudo inverse를 구할 수 있는 `numpy.linalg.pinv()`를 적용해야한다는 점입니다. 물론 matrix의 경우 정방 행렬이 아니더라도 동일하게 `I` 속성으로 값을 구할 수 있습니다. 😆
    ```python
    np.linalg.inv(A)
    # array([[-3.,  2.],
    #        [ 2., -1.]])

    C.I
    # array([[-3.,  2.],
    #        [ 2., -1.]])
    ```

- 특이값 분해
    + `numpy.linalg.svd()`라는 함수를 통해 구할 수 있습니다.


위에서 사용했던 `numpy`의 선형대수 관련 모듈은 향후에 더 자세하게 다루도록 하겠습니다.


#### 인덱싱

- 원소에 대한 접근
    ```python
    X = np.array([[51, 55], [14, 19], [0, 4]])
    print(X)

    print(f"0행 {X[0]}")                  # 0행 [51 55]
    print(f"(0, 1) 위치의 원소 {X[0][1]}")  # (0, 1) 위치의 원소 55
    ```

- for indexing
    ```python
    # 모든 row를 순서대로 출력
    for row in X:
        print(row)
    ```

- bool indexing
    ```python
    condition = X > 15   # 15 초과인 값만 subsetting하고자 할 경우
    print(condition)

    X[condition]
    ```


---

## 3. Linear Algebra Application

### 3.1. 성적 처리

| 이름 | 중간 | 기말 | 수행 |
| :-------: |:-------:| :-------:| :-------:|
| 학생1      | 100 | 50  | 90|
| 학생2      | 70      |   85  | 80|
| 학생3 | 45      |    75  | 100 |

위와 같은 1학기의 성적표가 있고, (**중간**, **기말**, **수행**)의 반영 비율이 (35%, 45%, 20%) 일때의 총점 계산을 선형대수적으로 처리해 보겠습니다. 우선 위의 점수 테이블과 반영 비율을 각각, 행렬과  벡터로 표기하면 아래와 같습니다. (간단한 예시이므로, 각 영역별 총점이 모두 100점이라 가정하겠습니다) 

$$
X = \begin{bmatrix} 
100 & 50 & 90 \\\\ 70 & 85 & 80 \\\\ 45 & 75 & 100
\end{bmatrix}
$$


$$\vec{p} =
\begin{bmatrix}
0.35 & 0.45 & 0.2 
\end{bmatrix} ^T$$

따라서 이렇게 표현된 두 행렬의 곱을 통해 손쉽게 원하는 결과를 얻을 수 있게되는 것입니다.

$$X \cdot \vec{p} =
\begin{bmatrix}
100 & 50 & 90 \\\\ 70 & 85 & 80 \\\\ 45 & 75 & 100
\end{bmatrix} \cdot
\begin{bmatrix}
0.35 \\\\ 0.45 \\\\ 0.2 
\end{bmatrix}
$$

```python
import numpy as np 

X = np.mat([[100, 50, 90],
            [70, 85, 80],
            [45, 75, 100]])

p = np.mat([0.35, 0.45, 0.2]).T

print(X * p)

# [[75.5 ] 학생1
#  [78.75] 학생2
#  [69.5 ]] 학생3

```

### 3.2. 기술 통계(Descriptive statistics)

> `기술 통계`는 정보 수집의 특징을 정량적으로 설명하거나 요약하는 요약 통계입니다[[1]](https://en.wikipedia.org/wiki/Descriptive_statistics#cite_note-1). 즉, 데이터를 요약, 설명하는데 초점이 맞추어져 있으며 다음과 같이 크게 2가지 기법이 있습니다.

1. 집중화 경향 (Central tendency): 데이터가 어떤 값에 집중되어 있는가?
    - 평균(Mean, Average)
2. 분산도(Variation): 데이터가 어떻게 퍼져 있는가?
    - 분산(Variance), 표준편차(Standard Deviation)

데이터의 갯수가 $n$개 이고, 데이터의 각 성분을 $d_i$로 표현할 때,

$$\text{Mean} = \cfrac{d_1 + d_2 + \cdots + d_n}{n} = \cfrac{\sum d_i}{n} = \bar{d}$$

$$\begin{aligned} \text{Variance} &= \cfrac{(d_1 - \bar{d})^2 + (d_2 - \bar{d})^2 + \cdots + (d_n - \bar{d})^2}{n - 1} \\ &= \cfrac{\sum (d_i - \bar{d})^2}{n - 1} = \sigma^2\end{aligned}$$

$$\text{Standard Deviation} = \sqrt{\sigma^2} = \sigma$$

그러면 머신러닝의 대표적인 예제 데이터 중 하나인 붓꽃(iris) 데이터에서 기술 통계량을 직접 구해보는 과정을 보여드리겠습니다.

```python
import numpy as np

data = np.loadtxt("data/iris.csv", delimiter=",", dtype=np.float32)

d = data[:, 1:5]

n = d.shape[0]
data_mean = d.sum(axis=0) / n
print(f"각 열의 평균: {data_mean}")

data_var = ((d - data_mean)**2 ).sum(axis=0) / (n - 1)
print(f"각 열의 분산: {data_var}")

data_std = data_var**(1/2)
print(f"각 열의 표준편차: {data_std}")
```

각 열의 평균: `[ 5.84333333  3.05733333  3.758       1.19933333]`

각 열의 분산: `[ 0.68569351  0.18997942  3.11627785  0.58100626]`

각 열의 표준편차: `[ 0.82806613  0.43586628  1.76529823  0.76223767]`



### 3.3. 기울기 하강법(Gradient Descent Method)

> `기울기 하강법`이란 어떤 함수의 최솟값을 찾기 위해 그 함수를 1차 미분해서 기울기를 구한 후 그 기울기를 따라 내려가면서 최솟값을 찾는 방법이다. -데이터 애널리틱스(이재식, 2020)-

{{< image src="/images/ml_from_scratch/1/Untitled 2.png" caption="그림3. 비용(Cost) 함수 또는 손실(Loss) 함수에서의 기울기 하강법 적용" alt="" height="1000px">}}

[출처: "[Coursera Machine Learning - 읽기 자료]](https://www.coursera.org/learn/machine-learning/supplement/2GnUg/gradient-descent)"]

 

마지막으로 소개해드릴 응용 방안 중 하나는 앞으로 상당히 중요하게 쓰이는 기울기 하강법(or 경사 하강법)입니다. 머신러닝에서는 목적으로 하는 Task(e.g., 예측, 분류 등)를 잘 수행하기 위해서 완벽한 답을 찾기 보다는 최적의, 충분한 답을 찾기 위해 노력합니다. 기울기 하강법은 이러한 목적에 부합한 정답의 근사치(Approximation)를 찾기 위해 답안을 수정해나가는 방법론 중 하나로 이해하시면 좋을 것 같습니다. 시리즈의 바로 다음 글 부터는 *그림3*과 같이 기울기 하강법을 통해 가중치를 수정해나가는 과정을 보여드릴 것이므로, 여기서는 기본적인 이해를 위해 간단한 예시만 다루어보도록 하겠습니다.

우선 이차 함수 $f(x) = x^2 -2x + 3$의 최솟값을 구하는 상황을 가정해보겠습니다. 우리는 여기서 식을 $f(x) = (x - 1)^2 + 2$ 로 변환할 수 있고, 따라서 최솟값이 $2$라는 것을 금방 구살 수 있습니다. 하지만 컴퓨터의 수치계산으로는 이것을 어떻게 구할 수 있을지가 직면한 문제상황이라고 할 수 있습니다. 기울기 하강법의 아이디어로 문제를 푼다면 무작위의 위치에서 시작하고, 해당 위치에서 기울기가 가파르다면 값을 크게 수정하고 기울기가 완만하다면 값을 작게 수정하는 전략으로 설명될 수 있습니다. 따라서 이를 식으로 표현하면 다음과 같습니다. 

$$x_{t+1} = x_t - \alpha \cdot \cfrac{df(x_t)}{dx_t} $$

위 식을 통해 알 수 있는 것은 새로운 값의 수정을 위해서 기울기를 얼마만큼 반영하여 수정할 것이고, 최솟값을 구하고자 하는 함수의 도함수를 알아야한다는 것입니다. 도함수의 경우 $f'(x) = g(x) = 2x - 2$로 비교적 쉽게 구할 수 있고, $\alpha = 0.3$으로 설정해보겠습니다. 

```python
def f(x):
    return x**2 - 2*x + 3

def g(x):
    return 2*x -2

x = np.array([10]) # 시작 위치 x=10
alpha = 0.3

for i in range(10):
    x = x - alpha * g(x)
    print(f"수정된 x: {x} \t 해당 함수값: {f(x)}") 

# 수정된 x: [4.6] 	 해당 함수값: [14.96]
# 수정된 x: [2.44] 	 해당 함수값: [4.0736]
# 수정된 x: [1.576] 	 해당 함수값: [2.331776]
# 수정된 x: [1.2304] 	 해당 함수값: [2.05308416]
# 수정된 x: [1.09216] 	 해당 함수값: [2.00849347]
# 수정된 x: [1.036864] 	 해당 함수값: [2.00135895]
# 수정된 x: [1.0147456] 	 해당 함수값: [2.00021743]
# 수정된 x: [1.00589824] 	 해당 함수값: [2.00003479]
# 수정된 x: [1.0023593] 	 해당 함수값: [2.00000557]
# 수정된 x: [1.00094372] 	 해당 함수값: [2.00000089]
```

이로써 최솟값을 만족하는 조건이 $x=1$ 일 경우이며, 값이 $2$로 근사해간다는 사실을 확인할 수 있었습니다. x 의 시작값이나 $\alpha$를 다양한 값으로 조절하며 실험을 해보더라도, 적당한 반복만 주어지게 되면 앞선 최솟값을 만족하는 조건으로 수렴한다는 사실을 확인하실 수 있습니다. 

# 마치며

항상 정리해야지하고 마음만 먹은 뒤에 코드만 대충 작성했던 부분을 드디어 한곳에 모아 정리하기 시작했다는 점에서 감회가 새로운 것 같습니다. 일단 시작을 했다는 점에서 절반(?)은 했다는 안도감도들지만, 시리즈의 첫글이자 오랜만에 작성하는 글이다보니 두서 없고 부족한 부분이 많기 때문에, 글에서 제가 오류를 범한 부분이나 추가적으로 필요한 부분에 대한 충고나 조언은 항상 감사히 받고 반영하도록 하겠습니다. 긴 글 읽어주셔서 감사합니다! 😂



# Reference

- Coursera Machine Learning
    - Week1: Linear Regression with One Variable
    - Week1: Linear Algebra Review
- Numpy
    - 밑바닥 부터 시작하는 딥러닝 Ch01 [[github]](https://github.com/WegraLee/deep-learning-from-scratch/tree/master/ch01)
    - numpy 공식 문서 quickstart [[doc]](https://numpy.org/doc/stable/user/quickstart.html)
- 응용 파트
    - 기술통계와 추리 통계란 무엇인가? [[tistory]](https://drhongdatanote.tistory.com/25)
