---
title: "[Hugo] Quick Start 1"
date: 2020-07-10T01:23:30+09:00

resources:
- name: featured-image
  src: https://gohugo.io/images/gohugoio-card.png
- name: featured-image-preview
  src: https://gohugo.io/images/gohugoio-card.png

tags: ["hugo", "quick-start"]
categories: ["Development"]

description: "This article shows the basic Markdown syntax and format."

# cover: "https://gohugo.io/images/gohugoio-card.png"
lightgallery: true

---

<!-- <img src="https://gohugo.io/images/gohugoio-card.png" /> -->

# Introduction

요즘은 정말 많은 블로그들이 운영되면서 양질의 정보들을 제공해주고 있습니다. 

한국에서도 예전부터 다음, 네이버 블로그, 그리고 티스토리 등을 통해서 다양한 정보들이 제공되어 왔습니다.

그리고 최근들어서는 github 블로그, Medium 등의 매체로 전세계적으로 더욱 양질의 정보들이 제공되고 있습니다.  

따라서 이 글에서는 github 블로그를 생성하도록 도와주는 많은 기술들 중, `Go` 언어 기반의 정적 웹사이트 생성기(Static Website Generator)를 소개해보고자 합니다.

Hugo Quick Start 글을 통해 OS별 설치, 주요 명령어, 블로그 프리뷰 및 생성 까지 알아가실 수 있습니다. git을 사용하기 때문에, 우선 사전에 git 환경을 구성해주시면 되겠습니다.

- `git`: https://git-scm.com/downloads


## 1. Hugo 설치

Hugo로 블로그를 생성하기 위한 첫번째 단계로 Hugo를 설치해야 합니다.

제가 작업하고 있는 환경은 [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)로 Linux 환경입니다. 

Unix 기반의 Linux와 MacOS의 경우 Hugo를 설치하는 방법이 상당히 간단합니다.


### 1.1. Ubuntu

```bash
sudo apt-get install hugo 
```

### 1.2. MacOS

```bash
brew install hugo
```


### 1.3. Windows

윈도우의 경우 [휴고](https://github.com/gohugoio/hugo/releases)에서 릴리즈 버전만 다운 받아서 C 드라이브 밑에 압축을 해제한 파일을 넣어주고, 환경 변수를 세팅해주면 됩니다.
예를 들면 저는 `C:\Hugo\bin`에 해당 파일들을 넣어주고 환경변수에 해당 폴더 경로를 추가하였습니다. 
이렇게 환경변수까지 설정해준다면, 윈도우에서도 **cmd**나 **powershell**을 띄워서 Hugo 명령어를 사용할 수 있습니다. 


<br /> 

이렇듯 위의 방법들로 자신의 OS에 맞게 Hugo를 설치해주셨다면, 다음의 명령어로 현재 Hugo Version을 확인하여 정상적으로 설치가 되었는지 확인해줍니다.

```bash
hugo version
```

![hugo version check](/images/hugo_quick_start/hugo_version.png)

## 2. Hugo 사이트 생성

이제 hugo의 사이트 생성 기능을 포함한 다양한 기능을 사용할 수 있습니다. 

### 2.1. `new site`



`new` 키워드로 할 수 있는 작업은 크게 두가지로, 새로운 블로그 프로젝트를 생성할 때 사용가능합니다.

```bash
hugo new site quickstart
```

다음의 명령어로 저희는 `quickstart`라는 폴더안에 새로운 블로그 프로젝트를 생성한 것입니다. 

해당 프로젝트는 처음보면 복잡한 구조로 되어 있는데, 향후 Hugo에 관한 자세한 글을 작성하여 설명드리도록 하겠습니다.

### 2.2. `new posts`

`new` 키워드는 또한 새로운 포스트를 만들때도 사용할 수 있습니다.

```bash
cd ./quickstart
hugo new posts/my-first-post.md
```

해당 명령어를 통해 저희의 프로젝트에서 `content/posts/my-frist-post.md`라는 [markdown](https://gist.github.com/ihoneymon/652be052a0727ad59601)) 문서가 생성되었습니다.


## 3. Hugo 테마 사용하기

지금의 상태로는 프로젝트를 실행하더라도 아무것도 보이지 않는 상태일 것입니다.

이는 현재 블로그를 어떻게 보여줄지에 대한 테마가 지정되지 않았기 때문입니다. 

`themes` 폴더에서 직접 테마를 개발하여 사용할 수도 있지만, 이미 잘 만들어진 테마들을 빠르게 적용해보는 것도 좋은 방법입니다.

> Hugo Theme List: https://themes.gohugo.io/

### 3.1. 테마 다운로드 

처음 적용해보시는 분들을 위해서 간단하면서도 아름다운 [Ananke](https://themes.gohugo.io/gohugo-theme-ananke/) 테마를 적용해보도록 하겠습니다.

다음과 같이 폴더에서 git 초기화를 해주신 다음, `clone`으로 받아오는 것이 아니라 사용하고자하는 테마를 `submodule`로 추가해주셔야 합니다.

```bash
git init
git submodule add https://github.com/budparr/gohugo-theme-ananke.git themes/ananke
```

뒷부분의 `themes/ananke`는 해당 테마를 프로젝트의 `themes` 폴더에 하위에 `ananke`라는 폴더를 만들어 저장하겠다는 의미입니다. 

### 3.2. 테마 `config.toml` 적용

테마의 다운로드가 완료되었다면 프로젝트 폴더 안에 있는 `config.toml`을 수정하여 테마가 적용되도록 해야합니다.

다음과 같은 초기 파일의 내용에 

```toml
baseURL = "http://example.org/"
languageCode = "en-us"
title = "My New Hugo Site"
```

`theme = ananke` (앞서 저장된 폴더이름)를 한줄 추가하면 됩니다.


## 4. Hugo 실행 및 생성

### 4.1. 블로그 프리뷰 
이제 드디어 블로그가 어떻게 생길지에 대해서 확인할 수 있게 되었습니다.

```bash
hugo server -D
```
해당 코드를 실행하면 블로그에 테마가 적용된 모습과 작성된 포스트들에 대해서 볼 수 있습니다.

![hugo server -D](/images/hugo_quick_start/hugo_server.png)

원래는 기본이 `localhost:1313` 혹은 `127.0.0.1:1313` 과 같이 1313 포트에서 실행돼야 하는데, 여러개를 실행할 경우 랜덤으로 남아있는 포트가 지정되게 됩니다.

혹시 theme를 가져왔는데 `hugo server -D` 가 실행되지 않는 경우가 간혹 있습니다. 
이럴때는 **hugo extended version**을 추가 설치하면 대부분의 문제가 해결됩니다.

### 4.2. 블로그 생성

앞선 명령어는 단지 자신의 로컬 환경에서 블로그가 어떻게 보여질지에 대한 피리뷰를 제공했다면, 다음의 명령어는 `html`, `css`, `javascript` 등으로 이루어진 실제 웹사이트를 생성해줍니다.

```bash
hugo -D
```

public 폴더가 새로 만들어지며 파일들이 생성되는데, 여기서 만들어진 파일을 호스팅할 github repository에 올리면 블로그가 호스팅되는 것입니다.

# Conclusion

이번 글을 통해 Hugo를 설치하는 과정부터 Github 배포를 위한 전단계까지의 과정에 대해서 알아보았습니다.

다음 Hugo 관련 글에서는 Github로 만들어진 웹사이트를 호스팅하는 방법에서부터 더 자세한 Hugo 구조에 대해서 다루어보도록 하겠습니다.

감사합니다.

# Reference

- [Quick Start | Hugo](https://gohugo.io/getting-started/quick-start/) 
