# Hugo 블로그 시작하기

: 2020년 07월 10일 안상호

> Hugo 기반 github 블로그 제작기


1. Introduction
2. Develop Desc 
3. Content Desc 
4. Reference 

## 1. Intorduction



## 2. Develop Desc 

기본적인 기능이 탄탄하기는 했지만, 저는 수학 수식과 코드를 이용해서 데이터분석과 프로그래밍을 중점적으로 포스팅할 수 있고 개성있는 블로그를 제작하고 싶었기에 다음과 같은 추가 요구사항이 존재했습니다.  

### Requirements

1. **KaTex** for Math Expression
2. **Fixed Page** for About Me, etc.
3. **Pygments** Code Highlighting to replace Highlight.js
4. **Subcategory** for classification
5. **Utterances** for Comments and Discussion
6. Some **Design** Features
  + Color Scheme
  + Logo and Thumbnail
  + Simple `Css`, `Java Script` Animation
  + Support `Dark Mode`  


## 3. Content Desc

블로그에서 다루고자하는 주요 주제는 크게 3가지로 *기계학습과 수학*(**M**achine Learning & **M**ath), *개발*(**D**evelop), 그리고 *디자인*(**D**esign) 입니다.  

일단 그 외의 분야는 기타 주제로 하여 포스팅 하되, 향후 주요하게 다룬다 싶으면 추가하고자 합니다. 

### Machine Learning & Math

- Deep Learning  
- Statistics for ML (with `R`)
- Linear Algebra for ML (with `Python`, `R`)
- Natural Language Processing, Computer Vision, Speech Recognition etc.. 
- Web Crawling을 활용한 데이터 수집

### Develop

- `Python` 백엔드 기반 웹서버
- `C#` 및 `XAML`을 활용한 윈도우앱
- `Golang` 기초 및 응용 
- `Web` 개발


### Design

- GIMP를 활용한 사진 편집
- InkScape를 활용한 그래픽 디자인
- UI/UX 작업 


## 4. Reference


![Hugo Hello Programmer Theme Screenshot](https://github.com/lubang/hugo-hello-programmer-theme/blob/master/images/screenshot.png)

## Base theme

제가 블로그를 만들 때 사용한 기본 테마는 [Lubang](https://blog.lulab.net/projects/2019-05-hugo-hello-programmer-theme-v2/) 님이 작성한  [Hugo-Hello-Programmer-Theme](https://themes.gohugo.io/hugo-hello-programmer-theme/) 로 주요 기능은 다음과 같습니다.  

- `Categories` and `Tags` Page
- `TOC` for each Post
- **Multilingual** Blog
- **Cover Image** for each Post


## Additional Features

기본적인 기능이 탄탄하기는 했지만, 저는 수학 수식과 코드를 이용해서 데이터분석과 프로그래밍을 중점적으로 포스팅할 수 있고 개성있는 블로그를 제작하고 싶었기에 다음과 같은 추가 요구사항이 존재했습니다.  

### 초기 요구사항

1. **KaTex** for Math Expression
2. **Fixed Page** for About Me, etc.
3. **Pygments** Code Highlighting to replace Highlight.js
4. **Subcategory** for classification
5. **Utterances** for Comments and Discussion
6. Some **Design** Features
  + Color Scheme
  + Logo and Thumbnail
  + Simple `Css`, `Java Script` Animation

### 추가된 기능

- `<img>` Modal Event
  + 이미지 클릭시 팝업 형태로 띄워줍니다
- `Subcategory`

## Update History

- **2019-07-20**
  + **Subcategory** Hierarchy
  + `Google Analytics` code
- **2019-07-18**
  + **Responsive** `Image modal` (`Desktop, Notebook, and Mobile)
  + Markdwon Image Resize and Align with `#center`
- **2019-07-17**
  + `Image Modal` Event (Multi & Dynamic)
- **2019-07-15**
  + `Image Modal` Event (Single & Static)
- **2019-07-12**
  + **Color Scheme** 변경
    - Primary: <font color="#374785">#374785</font>
    - Accent: <font color="#F76C6C">#F76C6C</font>
  + **Post Style** 변경
    - Shadow Effect
    - ETC
- **2019-07-11**
  + **pygments**를 이용한 code Highlighting
  + **KaTex**를 사용한 수학 수식
- **2019-07-10**
  + About 페이지 (fixed) 페이지 제작  

## Special Thanks To & References

사실 블로그 개설까지  이 테마 저 테마 적용하고 수정해보면서 상당히 많은 시행착오 ~~삽질~~ 를 겪으면서 점점 지쳐가기도 했지만, 또한 Hugo 블로그에 대한 많은 것을 알 수 있는 기회였습니다. 그래서 이 분들의 자료가 없었더라면 블로그를 개설하지 못했을텐데 많은 가르침을 주셔서 난관을 극복하게 해주신 몇몇 분들의 소스를 소개하도록 하겠습니다.  

- [ialy1595](https://ialy1595.github.io/post/blog-construct-2/)
  + Lubang님의 [Hello-Programmer-Theme](https://github.com/lubang/hugo-hello-programmer-theme)를 제가 가장 원하는 방식으로 커스터마이징 해주셨고, 제작기와 커스터마이징 포스트를 잘 정리하여 공유해 주셨습니다. 그래서 이러한 것들을 참조하여 제가 원하는데로 커스터마이징 해서 블로그를 배포할 수 있었습니다.
- [allgg](https://allgg.me/article/how-to-use-hugo-material-blog-theme/)
  + [Hugo Material Theme](https://github.com/digitalcraftsman/hugo-material-docs)라는 상당히 디자인적으로 호감이 갔던 테마를 알게 해주셨고, 이를 커스터마이징하셔서 이 소스를 가지고 이리 씨름하고 저리 씨름하며 Hugo 블로그에 대한 정말 많은 사항들을 배울 수 있었습니다.   

이 외에도 가장 기본적인 Hugo 블로그의 기본기를 알게 해준 [Hugo Docs](https://gohugo.io/documentation/)와 이와 관련된 강의를 제공해준 [Mike Dane](https://www.youtube.com/watch?v=qtIqKaDlqXo&list=PLLAZ4kZ9dFpOnyRlyS-liKL5ReHDcj4G3), 그리고 질문만 잘하면 항상 작동하는 답을 알려주었던 [Hugo Community](https://discourse.gohugo.io/), 그리고 Hugo에 대해 아무것도 모를 때 친절하게 한국어로서 내용을 알려준 [Golang Korean Community](https://golangkorea.github.io/series/hugo-introduction/)에 감사의 말씀을 올립니다.
